"""Check Kenward-Roger's tests and the F-tests of mixed models, and dream's use of them.

Compares, gene by gene, on data simulated from fixed seeds:

- ``fixed_effect_tests(df_method="kenward_roger")`` with lmerTest's ``summary`` and
  ``contest1D`` (``ddf = "Kenward-Roger"``) and pbkrtest's ``vcovAdj``;
- ``fixed_effect_f_test`` with pbkrtest's ``KRmodcomp`` and lmerTest's ``contestMD``
  (``ddf = "Kenward-Roger"`` and ``"Satterthwaite"``), for two contrasts and for three
  that span two dimensions;
- with observation weights, ``kenward_roger_weights="average"`` with pbkrtest on lme4
  fits whose weights are divided by their mean, as dream fits them, and
  ``"precision"`` with pbkrtest on the same model without weights, the values, fixed
  effects and random effects multiplied by the square roots of the weights (y* = W^1/2
  y, X* = W^1/2 X, ``(0 + sw | donor)``): the weighted model itself, whose covariance
  pbkrtest then builds from its matrices;
- variancePartition's ``dream`` with ``ddf = "Kenward-Roger"``, ``"adaptive"`` and
  ``"Satterthwaite"`` after ``voomWithDreamWeights``, with ``eBayes`` and ``topTable``,
  including its F-tests of several coefficients, against
  ``voom_with_dream_weights``, ``fixed_effect_tests`` and
  ``moderated_fixed_effect_tests``, on studies of 12 to 24 samples, the adaptive
  choice at 19 and 20 samples, and genes whose adjusted covariance is ill conditioned;
- ``dream_voom`` with its default, ``df_method="adaptive"``, and an F-test, on a cube
  whose cell types have 12 and 24 samples, against ``dream`` on each cell type's
  samples, features and library sizes.

Designs: one random intercept with a three-level time course and a donor covariate;
unbalanced donors, some with one sample; crossed and nested random intercepts; a design
simulated without donor variance, so that most fits lie on the boundary; and
observation weights like voom's with one and two random intercepts.

At given estimates: lme4 and lmerTest are evaluated once at CytePool's REML estimates
(passed as lme4's starting values with one evaluation allowed), where Kenward-Roger's
adjusted covariance, degrees of freedom and F-tests are closed forms, compared to the
exact tier (1e-10); lmerTest's Satterthwaite F-tests differentiate the deviance
numerically and are compared to 1e-5, as in ``validation/mixed_models.py``. At the
optimum, lme4 fits each gene from its own start with Powell's BOBYQA (minqa) and a trust
region shrunk to 1e-10, and the tests are compared to the close tier (1e-6); dream's
moderated tests too, the fits made with that optimiser.

Deliberate differences, printed as such: dream's p-values of an F-test use the total
degrees of freedom of the first coefficient of the first gene for every gene, and its
``F.std`` the number of coefficients rather than their rank; CytePool gives each gene
its own degrees of freedom. And dream decides from R's estimate of the condition number
(``kappa``) which genes fall back to Satterthwaite's method, CytePool from the exact
one: the two can decide differently within about 2% of the bound, 1e6.

The script also reports how the default, dream's adaptive choice, changes the results
of studies with fewer than 20 samples, which ``dream_voom`` tested with
Satterthwaite's method before it.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 with variancePartition 1.36.2, pbkrtest, lmerTest, lme4 1.1-x, minqa and
    jsonlite. Set ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/kenward_roger_and_f_tests.py
    PYTHONPATH=. python validation/kenward_roger_and_f_tests.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.differential_expression import dream_voom
from cytepool.normalization import filter_by_expression, library_sizes, tmm_factors
from cytepool.stats.distributions import f_sf
from cytepool.stats.dream import moderated_fixed_effect_tests, voom_with_dream_weights
from cytepool.stats.mixed_models import (
    fit_mixed_model,
    fixed_effect_f_test,
    fixed_effect_tests,
)

FIXTURES = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "kenward_roger_and_f_tests"
)
EXACT, CLOSE = 1e-10, 1e-6
SATTERTHWAITE = 1e-5  # lmerTest's numerical derivatives (validation/mixed_models.py)
N_GENES, N_FIXTURE = 24, 6
RELATIVE = "max rel. difference"
SCALED = "max diff. / max(abs, 1)"
P_VALUE = "max diff. / (p max(1, abs log p))"

R_SCRIPT = r"""
suppressPackageStartupMessages({library(variancePartition); library(lmerTest); library(pbkrtest); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
given <- lmerControl(optimizer = "nloptwrap", calc.derivs = FALSE, check.conv.singular = "ignore",
  restart_edge = FALSE, boundary.tol = 0, optCtrl = list(maxeval = 1))
tight_bobyqa <- function(par, fn, lower, upper, control) {
  res <- minqa::bobyqa(par, fn, lower = lower, upper = upper,
    control = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 1e5))
  list(par = res$par, fval = res$fval, conv = 0, message = res$msg)
}
tight <- lmerControl(optimizer = "tight_bobyqa", calc.derivs = FALSE, check.conv.singular = "ignore")
groups_of <- function(fit) sub("\\..*$", "", names(getME(fit, "theta")))
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("variancePartition", "limma", "lme4", "lmerTest", "pbkrtest", "minqa"),
  function(p) as.character(packageVersion(p))))))
for (name in names(input$fixed)) {
  case <- input$fixed[[name]]
  d <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  G <- nrow(case$y); p <- ncol(case$x)
  res <- list(adjusted = list(), coef = list(), contrast = list(), f = list(), optimum = list())
  for (g in seq_len(G)) {
    w <- if (is.null(case$w)) rep(1, nrow(d)) else case$w[g, ]
    if (case$variant == "precision") {  # the weighted model, its rows scaled
      d$y <- sqrt(w) * case$y[g, ]; d$X <- sqrt(w) * case$x; d$sw <- sqrt(w); d$ww <- 1
    } else {
      d$y <- case$y[g, ]; d$X <- case$x; d$sw <- 1; d$ww <- w / mean(w)
    }
    form <- as.formula(paste("y ~ 0 + X +", case$random))
    probe <- quiet(lmer(form, d, weights = ww, REML = TRUE, control = given))
    theta <- case$theta[g, match(groups_of(probe), case$components)]
    fit <- quiet(lmerTest::lmer(form, d, weights = ww, REML = TRUE, start = list(theta = theta), control = given))
    res$adjusted[[g]] <- as.matrix(vcovAdj(fit))
    res$coef[[g]] <- summary(fit, ddf = "Kenward-Roger")$coefficients[, c("Estimate", "Std. Error", "df", "t value", "Pr(>|t|)")]
    res$contrast[[g]] <- unlist(contest1D(fit, case$contrast, ddf = "Kenward-Roger")[1, ])
    res$f[[g]] <- lapply(case$f, function(L) {
      L <- matrix(L, ncol = p)
      c(unlist(KRmodcomp(fit, L)$test[1, c("stat", "ndf", "ddf", "F.scaling", "p.value")]),
        unlist(contestMD(fit, L, ddf = "Kenward-Roger")[1, c("F value", "NumDF", "DenDF", "Pr(>F)")]),
        unlist(contestMD(fit, L, ddf = "Satterthwaite")[1, c("F value", "NumDF", "DenDF", "Pr(>F)")]))
    })
    best <- quiet(lmerTest::lmer(form, d, weights = ww, REML = TRUE, control = tight))
    res$optimum[[g]] <- summary(best, ddf = "Kenward-Roger")$coefficients[, c("Std. Error", "df")]
  }
  out$fixed[[name]] <- res
}
for (name in names(input$dream)) {
  case <- input$dream[[name]]
  counts <- case$counts; data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  colnames(counts) <- rownames(data) <- paste0("s", seq_len(ncol(counts)))
  rownames(counts) <- paste0("g", seq_len(nrow(counts)))
  form <- as.formula(case$formula)
  v <- quiet(voomWithDreamWeights(counts, form, data, lib.size = case$lib_size, span = 0.5, control = tight))
  L <- if (length(case$contrasts)) quiet(makeContrastsDream(form, data, contrasts = unlist(case$contrasts))) else NULL
  r <- list(weights = v$weights)
  for (ddf in case$ddf) {
    fit <- quiet(dream(v, form, data, L = L, ddf = ddf, control = tight))
    e <- eBayes(fit)
    s <- list(names = colnames(fit$coefficients), df = fit$df.residual, se = fit$stdev.unscaled * fit$sigma,
      t = e$t, df_total = e$df.total, p = e$p.value, lods = e$lods,
      z = sapply(colnames(fit$coefficients), function(k) topTable(e, coef = k, number = Inf, sort.by = "none")$z.std))
    for (f in names(case$f_tests)) {
      top <- topTable(e, coef = case$f_tests[[f]], number = Inf, sort.by = "none")
      s$f[[f]] <- list(F = top$F, p = top$P.Value)
    }
    r[[ddf]] <- s
  }
  if (isTRUE(case$condition)) {  # the condition numbers that decide dream's fallback
    r$kappa <- sapply(seq_len(nrow(v$E)), function(g) {
      d <- data; d$yy <- v$E[g, ]; d$ww <- v$weights[g, ] / mean(v$weights[g, ])
      m <- quiet(lmerTest::lmer(update(form, yy ~ .), d, weights = ww, control = tight))
      kappa(as.matrix(vcovAdj(m)))
    })
  }
  out$dream[[name]] <- r
}
write_json(out, args[[2]], digits = NA, matrix = "rowmajor", auto_unbox = TRUE)
"""


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, ...]] = []
        self.failures = 0

    def check(
        self, check: str, reference: str, metric: str, value: float, tolerance: float
    ) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        result = "pass" if passed else "FAIL"
        self.rows.append(
            (check, reference, metric, f"{value:.1e}", f"{tolerance:.0e}", result)
        )

    def other(
        self,
        check: str,
        reference: str,
        metric: str,
        value: str,
        expected: bool | None = None,
    ) -> None:
        """An information row, or a deliberate difference that must be as expected."""
        self.failures += expected is False
        result = (
            "info"
            if expected is None
            else "deliberate difference"
            if expected
            else "FAIL"
        )
        self.rows.append((check, reference, metric, value, "n/a", result))

    def print(self) -> None:
        print("| check | reference | metric | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


def relative(actual: Any, desired: Any, floor: Any = 0.0) -> float:
    """Largest difference over max(|desired|, floor); infinite where NaN differs."""
    actual, desired = np.asarray(actual, float), np.asarray(desired, float)
    if actual.shape != desired.shape or not np.array_equal(
        np.isnan(actual), np.isnan(desired)
    ):
        return np.inf
    finite = np.isfinite(desired)
    scale = np.broadcast_to(np.fmax(np.abs(desired), floor), desired.shape)[finite]
    difference = np.abs(actual[finite] - desired[finite])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(difference == 0.0, 0.0, difference / scale)
    return float(np.max(ratios, initial=0.0))


def p_value_difference(actual: Any, desired: Any) -> float:
    """Difference relative to max(p, 1e-300) max(1, |log p|), the scale of log p."""
    p = np.fmax(np.asarray(desired, float), 1e-300)
    return relative(actual, desired, p * np.fmax(1.0, np.abs(np.log(p))))


def labels(prefix: str, codes: Any) -> list[str]:
    return [f"{prefix}{code + 1:02d}" for code in np.asarray(codes)]


# The tests of the fixed effects: pbkrtest and lmerTest at CytePool's estimates.


def simulate(rng: Any, design: Design, n_genes: int, random_sd: float) -> np.ndarray:
    """Values with gene-wise effects, residual SDs and random intercepts, whose SD is
    up to ``random_sd`` times the residual one and zero for a fifth of the genes."""
    x = design.matrix
    effects = rng.normal(0.0, 1.0, (x.shape[1], n_genes)) * (
        rng.random((x.shape[1], n_genes)) < 0.6
    )
    effects[0] = rng.uniform(2.0, 10.0, n_genes)
    sd = 0.3 * np.sqrt(4.0 / rng.chisquare(4.0, n_genes))
    y = x @ effects
    for effect in design.random:
        joined = [
            "\t".join(map(str, row))
            for row in zip(*(design.samples[g] for g in effect.grouping))
        ]
        codes = np.unique(joined, return_inverse=True)[1]
        scale = rng.uniform(0.0, random_sd, n_genes) * (rng.random(n_genes) > 0.2)
        y = y + rng.normal(0.0, 1.0, (codes.max() + 1, n_genes))[codes] * scale * sd
    return y + rng.normal(0.0, 1.0, y.shape) * sd


def fixed_cases(seed: int, n_genes: int) -> dict[str, dict[str, Any]]:
    """The designs, values, weights, contrasts and F-tests of the fixed effects' tests."""
    rng = np.random.default_rng(seed)
    donor = np.repeat(np.arange(6), 3)
    time_course = {
        "donor": labels("D", donor),
        "time": ["t0", "t1", "t2"] * 6,
        "age": np.round(rng.uniform(20.0, 80.0, 6), 1)[donor].tolist(),
    }
    sizes = [1, 2, 2, 3, 3, 4, 4, 5]
    unbalanced = {
        "donor": labels("D", np.repeat(np.arange(8), sizes)),
        "condition": rng.choice(["ctrl", "stim"], sum(sizes)).tolist(),
        "x": np.round(rng.normal(0.0, 1.0, sum(sizes)), 3).tolist(),
    }
    crossed = {
        "donor": labels("D", np.repeat(np.arange(8), 4)),
        "batch": labels("B", rng.integers(0, 4, 32)),
        "condition": ["ctrl", "ctrl", "stim", "stim"] * 8,
        "x": np.round(rng.normal(0.0, 1.0, 32), 3).tolist(),
    }
    nested = {
        "donor": labels("D", np.repeat(np.arange(4), 6)),
        "sample": labels("S", np.repeat(np.arange(3), 2).tolist() * 4),
        "time": np.repeat(["t0", "t1", "t2"], 2).tolist() * 4,
    }
    # F-tests: two contrasts, and three that span two dimensions.
    time_f = {
        "time": [[0, 1, 0, 0], [0, 0, 1, 0]],
        "pairs": [[0, 1, 0, 0], [0, 0, 1, 0], [0, -1, 1, 0]],
    }
    two_f = {"both": [[0, 1, 0], [0, 0, 1]]}
    specifications = [  # samples, formula, random, F-tests, a contrast, donor SD, weights
        (
            "time_course",
            time_course,
            "~ time + age",
            "(1 | donor)",
            time_f,
            [0, -1, 1, 0],
            2.0,
            False,
        ),
        (
            "unbalanced",
            unbalanced,
            "~ condition + x",
            "(1 | donor)",
            two_f,
            [0, 1, -1],
            2.0,
            False,
        ),
        (
            "crossed",
            crossed,
            "~ condition + x",
            "(1 | donor) + (1 | batch)",
            two_f,
            [0, 1, -1],
            2.0,
            False,
        ),
        (
            "nested",
            nested,
            "~ time",
            "(1 | donor/sample)",
            {"time": [[0, 1, 0], [0, 0, 1]]},
            [0, -1, 1],
            2.0,
            False,
        ),
        (
            "boundary",
            time_course,
            "~ time + age",
            "(1 | donor)",
            time_f,
            [0, -1, 1, 0],
            0.0,
            False,
        ),
        (
            "weights",
            time_course,
            "~ time + age",
            "(1 | donor)",
            time_f,
            [0, -1, 1, 0],
            2.0,
            True,
        ),
        (
            "weights_crossed",
            crossed,
            "~ condition + x",
            "(1 | donor) + (1 | batch)",
            two_f,
            [0, 1, -1],
            2.0,
            True,
        ),
    ]
    cases = {}
    for name, samples, formula, random, f, contrast, sd, weighted in specifications:
        y = simulate(rng, Design(samples, formula, random=random), n_genes, sd)
        w = None
        if weighted:  # positive weights that rise with expression, like voom's
            w = np.exp(rng.normal(0.0, 0.5, y.shape)) / (
                0.2 + 4.0 / (1.0 + np.exp(y - 5.0))
            )
            y = y + rng.normal(0.0, 0.3, y.shape) / np.sqrt(w)
        case = {
            "samples": samples,
            "formula": formula,
            "random": random,
            "f": f,
            "contrast": contrast,
            "y": y,
            "w": w,
            "variant": "average" if weighted else "none",
        }
        cases[name] = case
        if weighted:
            cases[f"{name}_precision"] = case | {"variant": "precision"}
    return cases


def run_fixed(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """CytePool's fits and tests of a case, and what R needs to evaluate lme4 there."""
    design = Design(case["samples"], case["formula"], random=case["random"])
    fit = fit_mixed_model(case["y"], design, weights=case["w"])
    options = {
        "df_method": "kenward_roger",
        "kenward_roger_weights": "precision"
        if case["variant"] == "precision"
        else "average",
    }
    ours = {
        "fit": fit,
        "kr": fixed_effect_tests(fit, **options),
        "contrast": fixed_effect_tests(
            fit, np.array(case["contrast"], float), **options
        ),
        "f": {
            key: (
                fixed_effect_f_test(fit, np.array(L, float).T, **options),
                fixed_effect_f_test(fit, np.array(L, float).T),
            )
            for key, L in case["f"].items()
        },
    }
    ratios = fit.variance_components / fit.residual_variance[:, None]
    if case["variant"] == "average":
        # lme4 fits the weights divided by their mean: its residual variance is that of
        # an observation of average weight.
        ratios = ratios * fit.mean_weights[:, None]
    to_r = {
        "samples": case["samples"],
        "random": case["random"].replace("(1 |", "(0 + sw |"),
        "x": design.matrix,
        "y": case["y"].T,
        "w": None if case["w"] is None else case["w"].T,
        "theta": np.sqrt(ratios),
        "components": list(fit.component_names),
        "contrast": case["contrast"],
        "f": case["f"],
        "variant": case["variant"],
    }
    return ours, to_r


def compare_fixed(
    name: str, ours: dict[str, Any], ref: dict[str, Any], results: Results
) -> None:
    kr, contrast, fit = ours["kr"], ours["contrast"], ours["fit"]
    adjusted = np.array(ref["adjusted"], float)
    coef = np.array(
        ref["coef"], float
    )  # genes x coefficients x (estimate, se, df, t, p)
    one = np.array(ref["contrast"], float)
    scale = np.max(np.abs(adjusted), axis=(1, 2))[:, None, None]
    # The tests of every coefficient use the adjusted covariance of the coefficients.
    results.check(
        f"{name}: adjusted covariance of the coefficients",
        "pbkrtest vcovAdj",
        "max diff. / max abs, per gene",
        relative(kr.covariance, adjusted, scale),
        EXACT,
    )
    results.check(
        f"{name}: Kenward-Roger t-tests of the coefficients and a contrast: estimates, standard "
        "errors, t",
        "lmerTest summary, contest1D",
        f"{SCALED} (estimates: of max(abs, se))",
        max(
            relative(kr.estimate, coef[..., 0], coef[..., 1]),
            relative(kr.standard_error, coef[..., 1]),
            relative(kr.t, coef[..., 3], 1.0),
            relative(contrast.estimate[:, 0], one[:, 0], one[:, 1]),
            relative(contrast.standard_error[:, 0], one[:, 1]),
            relative(contrast.t[:, 0], one[:, 3], 1.0),
        ),
        EXACT,
    )
    results.check(
        f"{name}: Kenward-Roger df and p-values",
        "lmerTest summary, contest1D",
        f"df: {RELATIVE}; p: {P_VALUE}",
        max(
            relative(kr.df, coef[..., 2]),
            p_value_difference(kr.p_value, coef[..., 4]),
            relative(contrast.df[:, 0], one[:, 2]),
            p_value_difference(contrast.p_value[:, 0], one[:, 4]),
        ),
        EXACT,
    )
    for key, (krf, satf) in ours["f"].items():
        # KRmodcomp's stat, ndf, ddf, F.scaling, p; contestMD's F, NumDF, DenDF, p by
        # Kenward-Roger, then by Satterthwaite.
        f = np.array([gene[key] for gene in ref["f"]], float)
        numerator = float(np.any(f[:, [1, 6, 10]] != krf.df1)) * np.inf
        results.check(
            f"{name}, F-test '{key}': Kenward-Roger F, df, scaling, p-value; lmerTest's the same",
            "pbkrtest KRmodcomp, lmerTest contestMD",
            f"{RELATIVE}; p: {P_VALUE}",
            max(
                relative(krf.f_statistic, f[:, 0]),
                relative(krf.df2, f[:, 2]),
                numerator,
                relative(krf.scaling, f[:, 3]),
                p_value_difference(krf.p_value, f[:, 4]),
                relative(f[:, 5], f[:, 0]),
                relative(f[:, 7], f[:, 2]),
            ),
            EXACT,
        )
        results.check(
            f"{name}, F-test '{key}': Satterthwaite F",
            "lmerTest contestMD",
            RELATIVE,
            relative(satf.f_statistic, f[:, 9]),
            EXACT,
        )
        results.check(
            f"{name}, F-test '{key}': Satterthwaite denominator df and p-value",
            "lmerTest contestMD",
            f"df: {RELATIVE}; p: {P_VALUE}",
            max(
                relative(satf.df2, f[:, 11]), p_value_difference(satf.p_value, f[:, 12])
            ),
            SATTERTHWAITE,
        )
    optimum = np.array(ref["optimum"], float)
    results.check(
        f"{name}: Kenward-Roger standard errors and df at lme4's optimum",
        "lmerTest summary",
        RELATIVE,
        max(
            relative(kr.standard_error, optimum[..., 0]),
            relative(kr.df, optimum[..., 1]),
        ),
        CLOSE,
    )
    results.other(
        f"{name}: fits with a variance at zero",
        "CytePool",
        "genes",
        f"{int(np.count_nonzero(fit.boundary.any(axis=1)))} of {fit.boundary.shape[0]}",
    )
    if kr.kenward_roger_weights == "average" and fit.weights is not None:
        # How the two treatments of the weights differ, and from the unadjusted
        # standard errors.
        precision = fixed_effect_tests(
            fit, df_method="kenward_roger", kenward_roger_weights="precision"
        )
        unadjusted = fixed_effect_tests(fit).standard_error
        se = kr.standard_error / precision.standard_error
        df = kr.df / precision.df
        results.other(
            f"{name}: every observation at the average weight against the precision "
            "weights",
            "CytePool",
            "standard errors, df: ratios, median (range); standard errors below the "
            "unadjusted ones: average, precision",
            f"{np.median(se):.3f} ({se.min():.3f} to {se.max():.3f}), {np.median(df):.3f} "
            f"({df.min():.3f} to {df.max():.3f}); "
            f"{np.mean(kr.standard_error < unadjusted):.0%}, "
            f"{np.mean(precision.standard_error < unadjusted * (1 - 1e-12)):.0%}",
        )


# dream: studies, and cubes for dream_voom.


def counts_of(
    seed: int, per_donor: list[int], n_genes: int, offset: float = 0.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Negative binomial counts with a donor effect, three time points and an age."""
    rng = np.random.default_rng(seed)
    donor = np.repeat(np.arange(len(per_donor)), per_donor)
    n = donor.size
    time = np.array(["t0", "t1", "t2"])[np.arange(n) % 3]
    age = np.round(rng.normal(size=n), 3)
    sd = rng.uniform(0.0, 0.6, n_genes) * (rng.random(n_genes) > 0.2)
    log_mean = (
        rng.normal(3.0, 1.5, n_genes)
        + (rng.normal(size=(len(per_donor), n_genes)) * sd)[donor]
    )
    for level in ("t1", "t2"):
        log_mean += np.outer(time == level, rng.normal(0.0, 0.5, n_genes))
    log_mean += np.outer(age, rng.normal(0.0, 0.2, n_genes))
    depth = rng.uniform(5e5, 2e6, n)
    counts = rng.negative_binomial(
        10, 10 / (10 + np.exp(log_mean) * (depth / depth.mean())[:, None])
    )
    return counts, {
        "donor": labels("D", donor),
        "time": time.tolist(),
        "age": (age + offset).tolist(),
    }


FORMULA, FIXED, RANDOM = "~ time + age + (1 | donor)", "~ time + age", "(1 | donor)"
CONTRASTS = {
    "t2_vs_t1": [0, -1, 1, 0],
    "(Intercept)": [1, 0, 0, 0],
    "timet1": [0, 1, 0, 0],
    "timet2": [0, 0, 1, 0],
    "age": [0, 0, 0, 1],
}
# Studies: samples per donor, and the values of dream's ddf to run.
STUDIES = {
    "n12": ([2] * 6, ["Kenward-Roger", "adaptive", "Satterthwaite"]),
    "n18": ([3] * 6, ["Kenward-Roger", "adaptive", "Satterthwaite"]),
    "n19": ([4, 4, 4, 4, 3], ["adaptive"]),
    "n20": ([4] * 5, ["adaptive"]),
    "n24": ([3] * 8, ["Kenward-Roger"]),
}


def study_cases() -> dict[str, dict[str, Any]]:
    cases = {}
    for seed, (name, (per_donor, ddf)) in enumerate(STUDIES.items(), start=20261101):
        counts, samples = counts_of(seed, per_donor, 40)
        cases[name] = {
            "counts": counts,
            "samples": samples,
            "formula": FORMULA,
            "ddf": ddf,
            "contrasts": {"t2_vs_t1": "timet2 - timet1"},
            "f_tests": {"time": ["timet1", "timet2"]},
            "lib_size": library_sizes(counts.astype(float)),
        }
    return cases


def ours_study(case: dict[str, Any], method: str) -> tuple[Any, Any, Any]:
    """CytePool's dream of a study: voom's weights, the moderated tests of every
    coefficient and contrast, and the moderated F-test of time."""
    design = Design(case["samples"], FIXED, random=RANDOM)
    voom = voom_with_dream_weights(case["counts"], design, span=0.5)
    fit = fit_mixed_model(voom.log_cpm, design, weights=voom.weights)
    tests = fixed_effect_tests(fit, CONTRASTS, df_method=method)
    time = fixed_effect_tests(
        fit, {k: CONTRASTS[k] for k in ("timet1", "timet2")}, df_method=method
    )
    return voom, moderated_fixed_effect_tests(tests), moderated_fixed_effect_tests(time)


def compare_f(
    label: str,
    f_value: Any,
    p_value: Any,
    dream_f: dict[str, Any],
    first_df: float,
    results: Results,
) -> None:
    """The moderated F-test against topTable's: the statistic, and the p-values that
    dream computes on one gene's degrees of freedom."""
    results.check(
        f"{label}: moderated F-statistic",
        "topTable F",
        RELATIVE,
        relative(f_value, dream_f["F"]),
        CLOSE,
    )
    dream_p = np.asarray(f_sf(np.asarray(dream_f["F"], float), 2.0, first_df))
    ratio = np.asarray(p_value) / np.asarray(dream_f["p"], float)
    results.other(
        f"{label}: p-values",
        "topTable P.Value",
        "dream's on the first gene's df.total, for every gene: rel. difference; CytePool's over dream's",
        f"{relative(dream_p, dream_f['p']):.0e}; {np.nanmin(ratio):.3g} to {np.nanmax(ratio):.3g}",
        relative(dream_p, dream_f["p"]) < EXACT,
    )


def compare_study(
    name: str, case: dict[str, Any], ref: dict[str, Any], results: Results
) -> list[str]:
    n = len(case["samples"]["donor"])
    voom, *_ = ours_study(case, "satterthwaite")
    results.check(
        f"{name}: precision weights",
        "voomWithDreamWeights",
        RELATIVE,
        relative(voom.weights, np.asarray(ref["weights"]).T),
        CLOSE,
    )
    found = {}
    for ddf in case["ddf"]:
        r = ref[ddf]
        method = (
            "kenward_roger"
            if ddf == "Kenward-Roger" or (ddf == "adaptive" and n < 20)
            else "satterthwaite"
        )
        _, moderated, time = ours_study(case, method)
        label = f"{name}, ddf = {ddf}" + (
            f" (CytePool: {method})" if ddf == "adaptive" else ""
        )
        columns = [r["names"].index(key) for key in CONTRASTS]
        pick = {
            key: np.asarray(r[key], float)[:, columns]
            for key in ("se", "df", "t", "lods", "df_total", "p", "z")
        }
        results.check(
            f"{label}: standard errors and df of the tests",
            "dream",
            RELATIVE,
            max(
                relative(moderated.tests.standard_error, pick["se"]),
                relative(moderated.tests.df, pick["df"]),
            ),
            CLOSE,
        )
        results.check(
            f"{label}: moderated t, B, z.std; df.total, p-values",
            "eBayes, topTable",
            f"t, B, z: {SCALED}; df.total: {RELATIVE}; p: {P_VALUE}",
            max(
                relative(moderated.t, pick["t"], 1.0),
                relative(moderated.lods, pick["lods"], 1.0),
                relative(moderated.z, pick["z"], 1.0),
                relative(moderated.df_total, pick["df_total"]),
                p_value_difference(moderated.p_value, pick["p"]),
            ),
            CLOSE,
        )
        first = float(np.asarray(r["df_total"], float)[0, r["names"].index("timet1")])
        compare_f(
            f"{label}, F-test of time",
            time.f_statistic,
            time.f_p_value,
            r["f"]["time"],
            first,
            results,
        )
        found[ddf] = moderated
    if "Satterthwaite" not in found:
        return []
    kr, satterthwaite = found["Kenward-Roger"], found["Satterthwaite"]
    ratio = kr.p_value[:, 0] / satterthwaite.p_value[:, 0]
    se = kr.tests.standard_error[:, 0] / satterthwaite.tests.standard_error[:, 0]
    return [
        f"study {name}, contrast t2_vs_t1",
        f"{np.nanmedian(ratio):.3g}; "
        f"{np.nanquantile(ratio, 0.05):.3g} to {np.nanquantile(ratio, 0.95):.3g}; "
        f"{np.nanmedian(se):.3g}, {np.mean(se < 1.0):.0%}",
    ]


def cube_cases() -> list[dict[str, Any]]:
    """Cubes for dream_voom: cell types with 12 and 24 samples left, and one of 15
    samples whose age, 32 years on average and 1 apart, gives condition numbers near
    1e6. And, for R, each cell type's samples, features and library sizes."""
    contrasts = {
        "t1_vs_t0": "timet1",
        "t2_vs_t0": "timet2",
        "t2_vs_t1": "timet2 - timet1",
    }
    cases = []
    for name, per_donor, offset, seeds in (
        ("time_course", [3] * 8, 0.0, (20261111, 20261112)),
        ("ill_conditioned", [3] * 5, 32.0, (20261113,)),
    ):
        layers = [counts_of(seed, per_donor, 60, offset)[0] for seed in seeds]
        samples = counts_of(seeds[0], per_donor, 60, offset)[1]
        n = len(samples["donor"])
        cells = np.random.default_rng(20261114).integers(20, 300, size=(n, len(seeds)))
        if len(seeds) == 2:
            cells[n // 2 :, 0] = 5  # "small": half the donors have too few cells
        samples["sample_id"] = [f"S{i:02d}" for i in range(n)]
        groups = ["small", "large"] if len(seeds) == 2 else ["cells"]
        cube = Cube(
            {"counts": np.stack(layers)},
            samples=samples,
            groups={"group_id": groups},
            features={"feature_id": [f"gene{i}" for i in range(60)]},
            n_cells=cells,
        )
        design = Design(
            cube.samples,
            FIXED,
            random=RANDOM,
            levels={"time": ["t0", "t1", "t2"]},
            contrasts=contrasts,
        )
        inputs = {}
        for g, group in enumerate(groups):
            values = np.asarray(cube.layer("counts", g), float)
            used = cells[:, g] >= 10
            keep = filter_by_expression(values[used], design=design.matrix[used])
            y = values[used][:, keep]
            inputs[group] = {
                "used": used,
                "keep": keep,
                "counts": y.T,
                "lib_size": library_sizes(y) * tmm_factors(y),
                "samples": {
                    k: np.asarray(samples[k])[used].tolist()
                    for k in ("donor", "time", "age")
                },
                "formula": FORMULA,
                "ddf": ["adaptive"],
                "condition": offset > 0,
                "contrasts": contrasts,
                "f_tests": {"time": ["t1_vs_t0", "t2_vs_t0"]},
            }
        cases.append({"name": name, "cube": cube, "design": design, "inputs": inputs})
    return cases


def compare_cube(
    case: dict[str, Any], refs: dict[str, Any], results: Results
) -> list[list[str]]:
    """dream_voom's default against dream on every cell type, and the change from
    Satterthwaite's method that the default makes."""
    changes = []
    # span = 0.5, voomWithDreamWeights' span.
    result = dream_voom(
        case["cube"],
        case["design"],
        span=0.5,
        f_tests={"time": ["t1_vs_t0", "t2_vs_t0"]},
    )
    before = dream_voom(
        case["cube"], case["design"], span=0.5, df_method="satterthwaite"
    )
    for entry in result.provenance["groups"]:
        group = entry["group_id"]
        given, ref = case["inputs"][group], refs[f"{case['name']}_{group}"]
        r = ref["adaptive"]
        label = f"cube {case['name']}, {group} ({len(entry['samples'])} samples: {entry['df_method']})"
        tested = np.asarray(result.table(group, "t1_vs_t0")["feature_id"])
        expected = np.array([f"gene{i}" for i in np.flatnonzero(given["keep"])])
        samples = np.asarray(case["design"].samples["sample_id"])[given["used"]]
        differences = np.count_nonzero(np.asarray(entry["samples"]) != samples)
        differences += (
            np.count_nonzero(tested != expected)
            if tested.size == expected.size
            else expected.size
        )
        results.check(
            f"{label}: samples used, features tested",
            "the rules of dream_voom",
            "differences",
            float(differences),
            0.0,
        )
        alike = np.ones(tested.size, bool)
        if given["condition"]:
            # dream decides from R's estimate of the condition number, CytePool from
            # the exact one: a gene near the bound can be decided differently.
            k = r["names"].index("t1_vs_t0")
            df = np.asarray(result.table(group, "t1_vs_t0")["df"])
            alike = np.abs(df - np.asarray(r["df_total"], float)[:, k]) <= 1e-4 * df
            kappa = np.asarray(ref["kappa"], float)
            near = np.abs(np.log(kappa / 1e6)) < np.log(1.02)
            results.other(
                f"{label}: genes whose Kenward-Roger tests fall back to Satterthwaite's",
                "dream, R's kappa of vcovAdj",
                "dream; CytePool; decided differently; all of those within 2% of 1e6",
                f"{int(np.count_nonzero(kappa > 1e6))}; {entry['features_satterthwaite_fallback']}; "
                f"{int(np.count_nonzero(~alike))}; {bool(np.all(near[~alike]))}",
                bool(np.all(near[~alike]))
                and entry["features_satterthwaite_fallback"] > 0,
            )
        worst = 0.0
        for name in case["design"].contrasts.names:
            table, k = result.table(group, name), r["names"].index(name)
            for column, key, floor in (
                ("t", "t", 1.0),
                ("b_statistic", "lods", 1.0),
                ("z_statistic", "z", 1.0),
                ("df", "df_total", 0.0),
            ):
                worst = max(
                    worst,
                    relative(
                        np.asarray(table[column])[alike],
                        np.asarray(r[key], float)[alike, k],
                        floor,
                    ),
                )
            worst = max(
                worst,
                p_value_difference(
                    np.asarray(table["p_value"])[alike],
                    np.asarray(r["p"], float)[alike, k],
                ),
            )
        alike_label = " (genes decided alike)" if given["condition"] else ""
        results.check(
            f"{label}: t, B, z; df; p-values of every contrast{alike_label}",
            "dream, eBayes, topTable",
            f"t, B, z: {SCALED}; df: {RELATIVE}; p: {P_VALUE}",
            worst,
            CLOSE,
        )
        if not given["condition"]:
            f_table = result.table(group, "time")
            first = float(
                np.asarray(r["df_total"], float)[0, r["names"].index("t1_vs_t0")]
            )
            compare_f(
                f"{label}, F-test of time",
                f_table["f_statistic"],
                f_table["p_value"],
                r["f"]["time"],
                first,
                results,
            )
        old, new = before.table(group, "t1_vs_t0"), result.table(group, "t1_vs_t0")
        ratio = np.asarray(new["p_value"]) / np.asarray(old["p_value"])
        significant = [
            int(np.nansum(np.asarray(t["adjusted_p_value"]) < 0.05)) for t in (old, new)
        ]
        changes.append(
            [
                f"cube {case['name']}, {group}, contrast t1_vs_t0",
                f"{np.nanmedian(ratio):.3g}; {np.nanquantile(ratio, 0.05):.3g} to "
                f"{np.nanquantile(ratio, 0.95):.3g}; {significant[0]} and {significant[1]} genes",
            ]
        )
    return changes


def run_r(payload: dict[str, Any]) -> dict[str, Any]:
    def plain(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                key: plain(item)
                for key, item in value.items()
                if key not in ("used", "keep")
            }
        return value

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tests.R").write_text(R_SCRIPT)
        (root / "input.json").write_text(json.dumps(plain(payload)))
        command = [
            os.environ.get("RSCRIPT", "Rscript"),
            "--vanilla",
            str(root / "tests.R"),
            str(root / "input.json"),
            str(root / "out.json"),
        ]
        subprocess.run(command, check=True)
        reference: dict[str, Any] = json.loads((root / "out.json").read_text())
        return reference


def write_fixtures(
    cases: dict[str, Any], refs: dict[str, Any], studies: dict[str, Any], cube: Cube
) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    genes = slice(0, N_FIXTURE)
    for name, case in cases.items():
        ref = refs["fixed"][name]
        arrays: dict[str, Any] = {
            "y": case["y"][:, genes],
            "adjusted": np.array(ref["adjusted"], float)[genes],
            "coef": np.array(ref["coef"], float)[genes],
            "contrast": np.array(ref["contrast"], float)[genes],
            "contrast_weights": np.array(case["contrast"], float),
        }
        if case["w"] is not None:
            arrays["weights"] = case["w"][:, genes]
        for key, L in case["f"].items():
            arrays[f"f {key}"] = np.array([gene[key] for gene in ref["f"]], float)[
                genes
            ]
            arrays[f"f_matrix {key}"] = np.array(L, float)
        np.savez_compressed(FIXTURES / f"{name}.npz", **arrays)
        meta = {key: case[key] for key in ("samples", "formula", "random", "variant")}
        (FIXTURES / f"{name}.json").write_text(json.dumps(meta) + "\n")
    for name in ("n12", "n20"):
        case, ref = studies[name], refs["dream"][name]
        arrays = {"counts": case["counts"]}
        for ddf in case["ddf"]:
            r = ref[ddf]
            columns = [r["names"].index(key) for key in CONTRASTS]
            for key in ("se", "df", "t", "lods", "df_total", "p", "z"):
                arrays[f"{ddf} {key}"] = np.asarray(r[key], float)[:, columns]
            arrays[f"{ddf} F"] = np.asarray(r["f"]["time"]["F"], float)
        np.savez_compressed(FIXTURES / f"study_{name}.npz", **arrays)
        (FIXTURES / f"study_{name}.json").write_text(json.dumps(case["samples"]) + "\n")
    arrays = {"counts": np.asarray(cube.layer("counts")), "n_cells": cube.n_cells}
    for group in ("small", "large"):
        r = refs["dream"][f"time_course_{group}"]["adaptive"]
        for name in ("t1_vs_t0", "t2_vs_t1"):
            k = r["names"].index(name)
            for key in ("t", "df_total", "p", "lods", "z"):
                arrays[f"{group} {name} {key}"] = np.asarray(r[key], float)[:, k]
        arrays[f"{group} time F"] = np.asarray(r["f"]["time"]["F"], float)
    np.savez_compressed(FIXTURES / "cube.npz", **arrays)
    (FIXTURES / "cube.json").write_text(
        json.dumps({k: cube.samples[k].tolist() for k in cube.samples.columns}) + "\n"
    )
    print(f"wrote fixtures to {FIXTURES}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = fixed_cases(20261031, N_GENES)
    fixed = {name: run_fixed(case) for name, case in cases.items()}
    studies = study_cases()
    cubes = cube_cases()
    dream_inputs = {
        name: case | {"counts": case["counts"].T} for name, case in studies.items()
    }
    for cube_case in cubes:
        dream_inputs |= {
            f"{cube_case['name']}_{g}": given
            for g, given in cube_case["inputs"].items()
        }
    refs = run_r(
        {
            "fixed": {name: to_r for name, (_, to_r) in fixed.items()},
            "dream": dream_inputs,
        }
    )
    print(
        "Reference: "
        + ", ".join(f"{name} {version}" for name, version in refs["versions"].items())
    )
    results = Results()
    for name, (ours, _) in fixed.items():
        compare_fixed(name, ours, refs["fixed"][name], results)
    changes = []
    for name, case in studies.items():
        changes += [compare_study(name, case, refs["dream"][name], results)]
    for cube_case in cubes:
        changes += compare_cube(cube_case, refs["dream"], results)
    for change in changes:
        if change:
            results.other(
                f"{change[0]}: the default against Satterthwaite's method",
                "CytePool",
                "p-values over Satterthwaite's: median; 5% to 95%; genes with adjusted p below 0.05 "
                "before and now"
                if change[0].startswith("cube")
                else "p-values over Satterthwaite's: median; 5% to 95%; standard errors over "
                "Satterthwaite's: median, share below 1",
                change[1],
            )
    results.print()
    print(f"\n{len(results.rows)} checks, {results.failures} failed")
    if arguments.write_fixtures:
        if results.failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(cases, refs, studies, cubes[0]["cube"])
    return 1 if results.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
