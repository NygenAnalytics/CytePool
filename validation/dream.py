"""Check dream against variancePartition: voom weights, moderated tests, a cube workflow.

Compares ``cytepool.stats.dream`` (``voom_with_dream_weights``,
``moderated_fixed_effect_tests``), the mixed models behind them (``fit_mixed_model``
by maximum likelihood and REML, with fitted values and approximate residual degrees
of freedom) and ``dream_voom`` with variancePartition's ``voomWithDreamWeights``,
``dream(ddf = "Satterthwaite")``, ``eBayes`` and ``topTable``, on counts simulated from
fixed seeds: a random donor intercept with a condition, a covariate and a contrast,
and ``eBayes`` with ``robust = TRUE`` and ``trend = TRUE``; crossed random intercepts
with sample prior weights; unbalanced donors, some with a single sample; and
``dream_voom`` on three cell types of a cube, with prior weights from the numbers of
cells, samples left out for having fewer than 10 cells, and a cell type left with one
sample per donor.

The optimum: variancePartition fits every gene with lme4, whose optimiser stops at
about 1e-6 of the optimum with its default tolerances. Here lme4 runs Powell's
BOBYQA from minqa (its ``"bobyqa"``) with the trust region shrunk to 1e-10, given by
name as lme4's ``optimizer`` argument allows and reporting convergence: at such
tolerances lme4 warns for some genes, and variancePartition drops a gene whose fit
warns. Estimates are compared to the close tier (1e-6), and
Satterthwaite's degrees of freedom, with the p-values and z-statistics that use them,
to 1e-5, because lmerTest differentiates the deviance numerically
(``validation/mixed_models.py`` checks CytePool's against the definition with 30
digits). At given estimates: lme4, evaluated once at CytePool's estimates, gives the
log-likelihood (maximum likelihood and REML) and the fitted values with the predicted
random effects, and variancePartition's ``rdf.merMod`` the approximate residual
degrees of freedom, closed forms compared to the exact tier (1e-10); lmerTest gives
Satterthwaite's degrees of freedom there, of maximum likelihood fits too.

In the cube workflow the features tested and the TMM factors are CytePool's
(``filter_by_expression`` and ``tmm_factors``, checked against edgeR by
``validation/differential_expression.py``), passed to R with the samples that
``limma_voom``'s rule keeps; the script checks that ``dream_voom`` used them.

Deliberate differences, printed as such: the default LOWESS span; z-statistics
beyond p = 1e-300; a feature that the fixed and random effects reproduce; a design
without random effects; and a cell type left with one sample per donor.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 with variancePartition 1.36.2, limma, lme4 1.1-x, lmerTest, minqa and
    jsonlite (variancePartition 1.36.2's ``dream`` fails for every gene with lme4
    2.0). Set ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/dream.py
    PYTHONPATH=. python validation/dream.py --write-fixtures
"""

import argparse
import dataclasses
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
from cytepool.stats.dream import moderated_fixed_effect_tests, voom_with_dream_weights
from cytepool.stats.mixed_models import fit_mixed_model, fixed_effect_tests
from cytepool.stats.multiple_testing import adjust_p_values

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "data" / "dream"
EXACT, CLOSE = 1e-10, 1e-6
SATTERTHWAITE = 1e-5  # lmerTest's numerical derivatives; see the docstring
N_GENES = 60
N_GIVEN = 20  # genes evaluated at CytePool's estimates
VARIANTS = {"default": {}, "robust": {"robust": True}, "trend": {"trend": True}}
RELATIVE = "max rel. difference"
SCALED = "max diff. / max(abs, 1)"
P_VALUE = "max diff. / (p max(1, abs log p))"

R_SCRIPT = r"""
suppressPackageStartupMessages({library(variancePartition); library(lme4); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
cases <- read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
# Powell's BOBYQA (minqa, lme4's "bobyqa") with its trust region shrunk to 1e-10, by
# name (lme4's optimizer argument), reporting convergence: lme4 warns for some genes
# at such tolerances, and variancePartition drops a gene whose fit warns. (nloptr's
# BOBYQA at tolerances near machine precision ended at worse points inside
# voomWithDreamWeights with crossed random effects.)
tight_bobyqa <- function(par, fn, lower, upper, control) {
  res <- minqa::bobyqa(par, fn, lower = lower, upper = upper,
    control = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 1e5))
  list(par = res$par, fval = res$fval, conv = 0, message = res$msg)
}
tight <- lmerControl(optimizer = "tight_bobyqa", calc.derivs = FALSE, check.conv.singular = "ignore")
# A single evaluation at the starting values: lme4 at given estimates.
given <- lmerControl(optimizer = "nloptwrap", calc.derivs = FALSE, check.conv.singular = "ignore",
  restart_edge = FALSE, boundary.tol = 0, optCtrl = list(maxeval = 1))
packages <- c("variancePartition", "limma", "lme4", "lmerTest", "pbkrtest", "Matrix", "RcppEigen", "minqa")
out <- list(versions = as.list(c(R = as.character(getRversion()),
  sapply(packages, function(p) as.character(packageVersion(p))))))
for (name in names(cases)) {
  case <- cases[[name]]
  counts <- case$counts  # features x samples
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  colnames(counts) <- rownames(data) <- paste0("s", seq_len(ncol(counts)))
  rownames(counts) <- paste0("g", seq_len(nrow(counts)))
  form <- as.formula(case$formula)
  w <- case$weights
  lib <- case$lib_size
  r <- list()
  if (isTRUE(case$fixed_only)) {
    v <- quiet(voomWithDreamWeights(counts, form, data, span = 0.5))
    r$class <- class(quiet(dream(v, as.formula(case$fixed), data)))[1]
    out[[name]] <- r; next
  }
  if (isTRUE(case$unreplicated)) {
    r$error <- tryCatch({
      v <- quiet(voomWithDreamWeights(counts, form, data, lib.size = lib, span = 0.5))
      quiet(dream(v, form, data, ddf = "Satterthwaite")); "none"
    }, error = function(e) conditionMessage(e))
    out[[name]] <- r; next
  }
  v <- quiet(voomWithDreamWeights(counts, form, data, lib.size = lib, span = 0.5, weights = w,
    rescaleWeightsAfter = !is.null(w), control = tight, hideErrorsInBackend = TRUE))
  r$kept <- match(rownames(v$E), rownames(counts))
  r$weights <- v$weights; r$sqrt_sd <- v$voom.xy$y
  if (!is.null(w)) r$default_weights <- quiet(voomWithDreamWeights(counts, form, data,
    lib.size = lib, span = 0.5, weights = w, control = tight))$weights
  L <- if (length(case$contrasts)) makeContrastsDream(form, data, contrasts = unlist(case$contrasts)) else NULL
  fit <- quiet(dream(v, form, data, L = L, ddf = "Satterthwaite", control = tight, hideErrorsInBackend = TRUE))
  r$dream_kept <- match(rownames(fit$coefficients), rownames(counts))
  r$names <- colnames(fit$coefficients)
  r$coefficients <- fit$coefficients; r$se <- fit$stdev.unscaled * fit$sigma
  r$df <- fit$df.residual; r$sigma <- fit$sigma; r$rdf <- fit$rdf
  for (variant in case$variants) {
    e <- eBayes(fit, robust = variant == "robust", trend = variant == "trend")
    top <- lapply(r$names, function(k) topTable(e, coef = k, number = Inf, sort.by = "none"))
    r[[variant]] <- list(s2_post = e$s2.post, df_prior = e$df.prior, s2_prior = e$s2.prior,
      t = e$t, df_total = e$df.total, p = e$p.value, lods = e$lods,
      z = sapply(top, function(x) x$z.std), adjusted = sapply(top, function(x) x$adj.P.Val),
      average = top[[1]]$AveExpr)
  }
  if (!is.null(case$extreme_coef)) {  # topTable's z.std where the p-value underflows
    k <- which(r$names == case$extreme_coef)
    fit$coefficients[1, k] <- fit$coefficients[1, k] * 1e13
    r$extreme_z <- topTable(eBayes(fit), coef = k, number = Inf, sort.by = "none")$z.std[1]
  }
  # lme4 at CytePool's estimates: the trend's maximum likelihood fits, dream's REML fits.
  for (kind in names(case$given)) {
    g <- case$given[[kind]]
    f <- as.formula(paste("yy", case$formula))
    res <- list(loglik = numeric(0), sigma2 = numeric(0), fitted = NULL, rdf = numeric(0), df = NULL, optimum = NULL)
    for (i in seq_len(nrow(g$y))) {
      d <- data; d$yy <- g$y[i, ]; d$ww <- g$w[i, ]
      m0 <- quiet(lmer(f, d, weights = ww, REML = g$reml, control = given))
      groups <- sub("\\.\\(Intercept\\)$", "", names(getME(m0, "theta")))
      m <- quiet(lmerTest::lmer(f, d, weights = ww, REML = g$reml, control = given,
        start = list(theta = g$theta[i, match(groups, case$components)])))
      res$df <- rbind(res$df, summary(m)$coefficients[, "df"])
      res$loglik <- c(res$loglik, as.numeric(logLik(m))); res$sigma2 <- c(res$sigma2, sigma(m)^2)
      res$fitted <- rbind(res$fitted, fitted(m)); res$rdf <- c(res$rdf, rdf.merMod(m))
      if (!g$reml) {  # the optimum, from lme4's own start
        m <- quiet(lmer(f, d, weights = ww, REML = FALSE, control = tight))
        res$optimum <- rbind(res$optimum, c(as.numeric(logLik(m)), sigma(m)^2))
      }
    }
    r[[paste0("given_", kind)]] <- res
  }
  out[[name]] <- r
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


def simulate(
    seed: int, per_donor: list[int], n_genes: int, batches: int = 0
) -> tuple[np.ndarray, dict[str, list[Any]]]:
    """Negative binomial counts with a donor effect, a condition and a covariate."""
    rng = np.random.default_rng(seed)
    donor = np.repeat(np.arange(len(per_donor)), per_donor)
    n = donor.size
    condition = np.where(np.arange(n) % 3 == 0, "ctrl", "stim")
    age = np.round(rng.normal(size=n), 3)
    sd = rng.uniform(0.0, 0.6, n_genes) * (rng.random(n_genes) > 0.2)
    log_mean = (
        rng.normal(3.0, 1.5, n_genes)
        + (rng.normal(size=(len(per_donor), n_genes)) * sd)[donor]
    )
    log_mean += np.outer(condition == "stim", rng.normal(0.0, 0.5, n_genes))
    log_mean += np.outer(age, rng.normal(0.0, 0.2, n_genes))
    samples: dict[str, list[Any]] = {
        "donor": [f"D{d}" for d in donor],
        "condition": condition.tolist(),
        "age": age.tolist(),
    }
    if batches:
        batch = rng.integers(batches, size=n)
        log_mean += (rng.normal(size=(batches, n_genes)) * 0.3)[batch]
        samples["batch"] = [f"B{b}" for b in batch]
    depth = rng.uniform(5e5, 2e6, n)
    return rng.negative_binomial(
        10, 10 / (10 + np.exp(log_mean) * (depth / depth.mean())[:, None])
    ), samples


def stats_cases() -> dict[str, dict[str, Any]]:
    """The cases of the stats functions: name, samples per donor, random effects, batches,
    prior weights (none, for samples, or for every observation) and eBayes variants."""
    cases = {}
    specifications = [
        ("one_intercept", [3] * 10, "(1 | donor)", 0, None, list(VARIANTS)),
        ("crossed", [4] * 8, "(1 | donor) + (1 | batch)", 3, "samples", ["default"]),
        (
            "unbalanced",
            [1, 2, 3, 4, 1, 2, 3, 4, 2, 3],
            "(1 | donor)",
            0,
            "observations",
            ["default"],
        ),
    ]
    for seed, (name, per_donor, random, batches, weighted, variants) in enumerate(
        specifications, start=20261025
    ):
        counts, samples = simulate(seed, per_donor, N_GENES, batches)
        rng = np.random.default_rng(seed + 100)
        if name == "one_intercept":  # two hypervariable genes, for robust = TRUE
            counts[:, :2] = np.round(
                counts[:, :2] * np.exp(rng.normal(0.0, 1.5, (counts.shape[0], 2)))
            )
        shape = {None: None, "samples": counts.shape[0], "observations": counts.shape}[
            weighted
        ]
        cases[name] = {
            "counts": counts,
            "samples": samples,
            "fixed": "~ condition + age",
            "random": random,
            "formula": f"~ condition + age + {random}",
            "contrasts": {"stim_minus_age": "conditionstim - age"},
            "variants": variants,
            "weights": None if shape is None else rng.uniform(20, 400, shape),
        }
    cases["one_intercept"]["extreme_coef"] = "conditionstim"
    return cases


def design_of(case: dict[str, Any]) -> Design:
    return Design(case["samples"], case["fixed"], random=case["random"])


def scaled_prior(weights: np.ndarray, n_samples: int) -> np.ndarray:
    """Prior weights with each feature's divided by their mean, as a samples x 1 or
    samples x features array."""
    return np.reshape(weights / np.mean(weights, axis=0), (n_samples, -1))


def add_given_fits(case: dict[str, Any], voom: Any) -> None:
    """Fits for lme4 to evaluate: the trend's (maximum likelihood) and dream's (REML)."""
    design, genes = design_of(case), np.arange(N_GIVEN)
    n = voom.log_cpm.shape[0]
    prior = (
        np.ones((n, 1)) if case["weights"] is None else scaled_prior(case["weights"], n)
    )
    case["components"] = [effect.group for effect in design.random]
    case["fits"], case["given"] = {}, {}
    for kind, weights in (
        ("ml", np.repeat(prior, N_GIVEN, axis=1)),
        ("reml", voom.weights[:, genes]),
    ):
        fit = fit_mixed_model(
            voom.log_cpm[:, genes], design, weights=weights, method=kind
        )
        theta = np.sqrt(fit.variance_components / fit.residual_variance[:, None])
        case["fits"][kind] = fit
        case["given"][kind] = {
            "y": voom.log_cpm[:, genes].T,
            "w": weights.T,
            "reml": kind == "reml",
            "theta": theta,
        }


def compare_moderation(
    label: str, ours: dict[str, np.ndarray], r: dict[str, Any], k: Any, results: Results
) -> None:
    """The moderated statistics of one fit (column k of R's arrays) against eBayes."""
    reference = {
        key: np.asarray(r[key])[:, k]
        for key in ("t", "lods", "df_total", "p", "adjusted", "z")
    }
    results.check(
        f"{label}: moderated t, B",
        "eBayes",
        SCALED,
        max(
            relative(ours["t"], reference["t"], 1.0),
            relative(ours["lods"], reference["lods"], 1.0),
        ),
        CLOSE,
    )
    results.check(
        f"{label}: df.total, p-values, adj.P.Val, z.std",
        "eBayes, topTable",
        f"df.total: {RELATIVE}; p, adj.P.Val: {P_VALUE}; z.std: {SCALED}",
        max(
            relative(ours["df_total"], reference["df_total"]),
            p_value_difference(ours["p"], reference["p"]),
            p_value_difference(ours["adjusted"], reference["adjusted"]),
            relative(ours["z"], reference["z"], 1.0),
        ),
        SATTERTHWAITE,
    )


def compare_stats(
    name: str, case: dict[str, Any], ref: dict[str, Any], results: Results
) -> None:
    design = design_of(case)
    voom = voom_with_dream_weights(
        case["counts"], design, span=0.5, weights=case["weights"]
    )
    weights = [
        relative(voom.weights, np.asarray(ref["weights"]).T),
        relative(voom.sqrt_sd, ref["sqrt_sd"]),
    ]
    if (
        case["weights"] is not None
    ):  # rescaleWeightsAfter = FALSE gives the trend's weights alone
        alone = voom.weights / scaled_prior(case["weights"], voom.weights.shape[0])
        weights.append(relative(alone, np.asarray(ref["default_weights"]).T))
    results.check(
        f"{name}: precision weights, square roots of the trend's SDs",
        "voomWithDreamWeights",
        f"weights (also the trend's alone), voom.xy y: {RELATIVE}",
        max(weights),
        CLOSE,
    )
    fit = fit_mixed_model(voom.log_cpm, design, weights=voom.weights)
    names, n = list(ref["names"]), fit.residual_df.size
    matrix = np.zeros((len(design.columns), len(names)))
    for j, column in enumerate(names):
        for sign, term in zip(
            (1.0, -1.0), case["contrasts"].get(column, column).split(" - ")
        ):
            matrix[design.columns.index(term), j] = sign
    tests = fixed_effect_tests(fit, matrix, names=names)
    se = np.asarray(ref["se"])
    results.check(
        f"{name}: coefficients, a contrast, standard errors",
        "dream",
        "max diff. / max(abs, se)",
        max(
            relative(tests.estimate, ref["coefficients"], se),
            relative(tests.standard_error, se),
        ),
        CLOSE,
    )
    results.check(
        f"{name}: Satterthwaite df",
        "dream",
        f"df.residual: {RELATIVE}",
        relative(tests.df, ref["df"]),
        SATTERTHWAITE,
    )
    for variant in case["variants"]:
        mod, r = moderated_fixed_effect_tests(tests, **VARIANTS[variant]), ref[variant]
        label = name + (
            ": eBayes" if variant == "default" else f": eBayes {variant} = TRUE"
        )
        if variant == "default":
            results.check(
                f"{name}: residual SD at the average weight, rdf",
                "dream",
                f"sigma, rdf: {RELATIVE}",
                max(
                    relative(
                        np.sqrt(mod.residual_variance_at_mean_weight), ref["sigma"]
                    ),
                    relative(mod.residual_df, ref["rdf"]),
                ),
                CLOSE,
            )
        each = [
            np.broadcast_to(np.asarray(value, float), (n,))
            for value in (
                mod.prior_df,
                r["df_prior"],
                mod.prior_variance,
                r["s2_prior"],
            )
        ]
        results.check(
            f"{label}: prior and posterior",
            "eBayes",
            f"df.prior, s2.prior, s2.post: {RELATIVE}",
            max(
                relative(each[0], each[1]),
                relative(each[2], each[3]),
                relative(mod.posterior_variance, r["s2_post"]),
            ),
            CLOSE,
        )
        ours = {
            "t": mod.t,
            "lods": mod.lods,
            "df_total": mod.df_total,
            "p": mod.p_value,
            "adjusted": adjust_p_values(mod.p_value, "bh", axis=0),
            "z": mod.z,
        }
        compare_moderation(label, ours, r, slice(None), results)
        results.other(
            f"{label}: prior df, estimator",
            "CytePool",
            "median, smallest, estimator",
            f"{np.median(mod.prior_df):.4g}, {np.min(mod.prior_df):.4g}, {mod.prior_estimator}",
        )
    if "extreme_z" in ref:
        k = names.index(case["extreme_coef"])
        estimate = tests.estimate.copy()
        estimate[0, k] *= 1e13
        z = moderated_fixed_effect_tests(
            dataclasses.replace(tests, estimate=estimate)
        ).z[0, k]
        results.other(
            f"{name}: z-statistic beyond p = 1e-300",
            "topTable",
            "z.std; CytePool z",
            f"{ref['extreme_z']:.4g}; {z:.4g}",
            abs(ref["extreme_z"]) < 37.1 < abs(z),
        )
    for kind, given in case.get("fits", {}).items():
        got, what = (
            ref[f"given_{kind}"],
            f"{name}: {'maximum likelihood' if kind == 'ml' else 'REML'}",
        )
        results.check(
            f"{what} at CytePool's estimates",
            "lme4, rdf.merMod",
            f"logLik, sigma^2, rdf: {RELATIVE}; fitted: max diff. / max abs",
            max(
                relative(given.log_likelihood, got["loglik"]),
                relative(given.residual_variance, got["sigma2"]),
                relative(given.residual_df, got["rdf"]),
                relative(
                    given.fitted_values,
                    np.asarray(got["fitted"]).T,
                    np.abs(given.fitted_values).max(),
                ),
            ),
            EXACT,
        )
        results.check(
            f"{what} at CytePool's estimates: Satterthwaite df",
            "lmerTest",
            f"summary df: {RELATIVE}",
            relative(fixed_effect_tests(given).df, got["df"]),
            SATTERTHWAITE,
        )
        if kind == "ml":
            optimum = np.asarray(got["optimum"])
            results.check(
                f"{what} optimum",
                "lme4",
                f"logLik, sigma^2: {RELATIVE}",
                max(
                    relative(given.log_likelihood, optimum[:, 0]),
                    relative(given.residual_variance, optimum[:, 1]),
                ),
                CLOSE,
            )
            below = np.count_nonzero(
                given.log_likelihood < optimum[:, 0] - EXACT * np.abs(optimum[:, 0])
            )
            results.check(
                f"{what}: log-likelihood below lme4's maximum",
                "lme4",
                "genes",
                float(below),
                0.0,
            )


def cube_case() -> tuple[Cube, Design, dict[str, dict[str, Any]]]:
    """Three cell types of 8 donors x 2 conditions, in which few cells leave samples out,
    and each group's samples (limma_voom's rule), features and library sizes, for R."""
    rng = np.random.default_rng(20261028)
    layers = []
    for seed in rng.integers(1 << 30, size=3):
        counts, samples = simulate(int(seed), [2] * 8, N_GENES)
        layers.append(counts)
    samples["condition"] = ["ctrl", "stim"] * 8
    samples["sample_id"] = [
        f"{d}_{c}" for d, c in zip(samples["donor"], samples["condition"])
    ]
    n_cells = rng.integers(20, 300, size=(16, 3))
    n_cells[[2, 3, 6], 1] = [4, 7, 9]  # NK: donor D1 has no sample left, D3 one
    # rare: three donors keep one sample each, which leaves residual df but cannot
    # separate the donor variance from the residual variance.
    n_cells[:, 2] = rng.integers(0, 10, size=16)
    n_cells[[0, 3, 4], 2] = [40, 60, 80]
    counts = np.stack(layers)
    counts[n_cells.T == 0] = 0  # a sample without cells in a group has no counts there
    cube = Cube(
        {"counts": counts},
        samples=samples,
        groups={"group_id": ["B", "NK", "rare"]},
        features={"feature_id": [f"gene{i}" for i in range(N_GENES)]},
        n_cells=n_cells,
    )
    design = Design(
        cube.samples,
        "~ condition",
        random="(1 | donor)",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )
    inputs = {}
    for g, group in enumerate(cube.groups["group_id"].tolist()):
        values = np.asarray(cube.layer("counts", g), float)
        used = (cube.n_cells[:, g] >= 10) & (values.sum(axis=1) > 0)
        keep = filter_by_expression(values[used], design=design.matrix[used])
        y = values[used][:, keep]
        donors = np.asarray(design.samples["donor"])[used].tolist()
        inputs[group] = {
            "used": used,
            "keep": keep,
            "counts": y.T,
            "lib_size": library_sizes(y) * tmm_factors(y),
            "samples": {
                "donor": donors,
                "condition": np.asarray(design.samples["condition"])[used].tolist(),
            },
            "weights": cube.n_cells[used, g].astype(float),
            "formula": "~ condition + (1 | donor)",
            "contrasts": {},
            "variants": ["default", "trend"],
            "unreplicated": len(set(donors)) == len(donors),
        }
    return cube, design, inputs


def compare_cube(
    cube: Cube,
    design: Design,
    inputs: dict[str, Any],
    refs: dict[str, Any],
    results: Results,
) -> None:
    for variant in ("default", "trend"):
        result = dream_voom(
            cube, design, prior_weights="n_cells", span=0.5, trend=variant == "trend"
        )
        for group, given in inputs.items():
            ref, label = (
                refs[f"cube_{group}"],
                f"cube, {group}" + (", trend = TRUE" if variant == "trend" else ""),
            )
            status = result.summary["status"][result.groups.index(group)]
            if given["unreplicated"]:
                if variant == "default":
                    results.other(
                        f"cube, {group}: one sample per donor left",
                        "voomWithDreamWeights",
                        "error; CytePool's status",
                        f"{ref['error'].splitlines()[0][:50]}; {status}",
                        status == "random_effects_not_estimable",
                    )
                continue
            table = result.table(group, "stim_vs_ctrl")
            rows = result.samples["group_id"] == group
            expected = np.array([f"gene{i}" for i in np.flatnonzero(given["keep"])])
            tested = np.asarray(table["feature_id"])
            differences = np.count_nonzero(
                result.samples["used"][rows] != given["used"]
            )
            differences += (
                np.count_nonzero(tested != expected)
                if tested.size == expected.size
                else expected.size
            )
            if variant == "default":
                results.check(
                    f"{label}: samples used, features tested",
                    "the rules of limma_voom",
                    "differences",
                    float(differences),
                    0.0,
                )
            k, r = list(ref["names"]).index("conditionstim"), ref[variant]
            results.check(
                f"{label}: logFC, AveExpr",
                "topTable",
                SCALED,
                max(
                    relative(
                        table["log2_fold_change"],
                        np.asarray(ref["coefficients"])[:, k],
                        1.0,
                    ),
                    relative(table["average_log2_cpm"], r["average"], 1.0),
                ),
                CLOSE,
            )
            ours = {
                "t": table["t"],
                "lods": table["b_statistic"],
                "df_total": table["df"],
                "p": table["p_value"],
                "adjusted": table["adjusted_p_value"],
                "z": table["z_statistic"],
            }
            compare_moderation(label, ours, r, k, results)


def deviation_cases() -> dict[str, dict[str, Any]]:
    counts, samples = simulate(20261029, [2] * 6, 30)
    # Gene 0 is constant within donors, whose two samples have equal library sizes:
    # the second's counts of the other genes are the first's, rolled by one gene.
    counts[1::2, 1:] = np.roll(counts[0::2, 1:], 1, axis=1)
    counts[:, 0] = np.repeat(np.arange(6) * 7 + 20, 2)
    base = {
        "counts": counts,
        "samples": samples,
        "fixed": "~ condition",
        "random": "(1 | donor)",
        "formula": "~ condition + (1 | donor)",
        "weights": None,
        "contrasts": {},
        "variants": ["default"],
    }
    return {"reproduced": base, "fixed_only": base | {"fixed_only": True}}


def write_fixtures(
    stats: dict[str, Any], refs: dict[str, Any], cube: Cube, design: Design
) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, case in stats.items():
        ref = refs[name]
        arrays = {
            key: np.asarray(ref[key], float)
            for key in ("coefficients", "se", "df", "sigma", "rdf", "sqrt_sd")
        }
        arrays |= {
            "counts": case["counts"],
            "weights": np.asarray(ref["weights"]).T,
            "names": np.array(ref["names"]),
        }
        if case["weights"] is not None:
            arrays["prior_weights"] = case["weights"]
        for variant in case["variants"]:
            arrays |= {
                f"{variant} {key}": np.asarray(value, float)
                for key, value in ref[variant].items()
            }
        for kind in case.get("given", {}):  # lme4 and lmerTest at CytePool's estimates
            got = ref[f"given_{kind}"]
            arrays |= {
                f"given_{kind} {key}": np.asarray(got[key], float)
                for key in ("loglik", "sigma2", "rdf", "df", "fitted")
            }
        np.savez_compressed(FIXTURES / f"{name}.npz", **arrays)
        (FIXTURES / f"{name}.json").write_text(json.dumps(case["samples"]) + "\n")
    arrays = {"counts": np.asarray(cube.layer("counts")), "n_cells": cube.n_cells}
    for group in ("B", "NK"):
        r = refs[f"cube_{group}"]
        k = list(r["names"]).index("conditionstim")
        arrays[f"{group} logFC"] = np.asarray(r["coefficients"])[:, k]
        for variant in ("default", "trend"):
            for key in ("t", "df_total", "p", "lods", "z", "adjusted"):
                arrays[f"{group} {variant} {key}"] = np.asarray(r[variant][key])[:, k]
            arrays[f"{group} {variant} average"] = np.asarray(r[variant]["average"])
    np.savez_compressed(FIXTURES / "cube.npz", **arrays)
    samples = {key: design.samples[key].tolist() for key in design.samples.columns}
    (FIXTURES / "cube.json").write_text(json.dumps(samples) + "\n")


def run_r(cases: dict[str, Any]) -> dict[str, Any]:
    def plain(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                key: plain(item)
                for key, item in value.items()
                if key not in ("fits", "used", "keep")
            }
        return value

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "dream.R").write_text(R_SCRIPT)
        (root / "cases.json").write_text(json.dumps(plain(cases)))
        command = [
            os.environ.get("RSCRIPT", "Rscript"),
            "--vanilla",
            str(root / "dream.R"),
            str(root / "cases.json"),
            str(root / "out.json"),
        ]
        subprocess.run(command, check=True)
        reference: dict[str, Any] = json.loads((root / "out.json").read_text())
        return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    stats = stats_cases()
    for name in ("one_intercept", "crossed"):
        case = stats[name]
        add_given_fits(
            case,
            voom_with_dream_weights(
                case["counts"], design_of(case), span=0.5, weights=case["weights"]
            ),
        )
    cube, design, inputs = cube_case()
    deviations = deviation_cases()
    # R takes features x samples, as counts and observation weights are in R.
    to_r = {
        name: case
        | {
            "counts": case["counts"].T,
            "weights": None if case["weights"] is None else case["weights"].T,
        }
        for name, case in (stats | deviations).items()
    }
    refs = run_r(to_r | {f"cube_{group}": value for group, value in inputs.items()})
    print(
        "Reference: "
        + ", ".join(f"{name} {version}" for name, version in refs["versions"].items())
    )
    results = Results()
    for name, case in stats.items():
        compare_stats(name, case, refs[name], results)
    compare_cube(cube, design, inputs, refs, results)

    case = stats["one_intercept"]
    adaptive = voom_with_dream_weights(case["counts"], design_of(case))
    difference = relative(
        adaptive.weights, np.asarray(refs["one_intercept"]["weights"]).T
    )
    results.other(
        "the default span",
        "voomWithDreamWeights",
        "span 0.5: CytePool's span; max rel. difference of the weights",
        f"{adaptive.span:.3g}; {difference:.1e}",
        adaptive.span != 0.5,
    )
    case, ref = deviations["reproduced"], refs["reproduced"]
    voom = voom_with_dream_weights(case["counts"], design_of(case), span=0.5)
    t = fixed_effect_tests(
        fit_mixed_model(voom.log_cpm, design_of(case), weights=voom.weights)
    ).t[0, 1]
    in_r = (
        f"{ref['sqrt_sd'][ref['kept'].index(1)]:.2g}" if 1 in ref["kept"] else "dropped"
    )
    results.other(
        "a feature the fixed and random effects reproduce",
        "voomWithDreamWeights, dream",
        "sqrt SD in the trend, dream keeps it; CytePool: sqrt SD, t",
        f"{in_r}, {1 in ref['dream_kept']}; {voom.sqrt_sd[0]:.2g}, {t:.2g}",
        bool(np.isnan(voom.sqrt_sd[0]) and np.isnan(t)),
    )
    try:
        dream_voom(cube, Design(cube.samples, "~ condition"))
        outcome = "fitted"
    except ValueError:
        outcome = "refused"
    results.other(
        "a design without random effects",
        "dream",
        "class of the fit; CytePool",
        f"{refs['fixed_only']['class']}; {outcome}",
        outcome == "refused",
    )
    results.print()
    print(f"\n{len(results.rows)} checks, {results.failures} failed")
    if arguments.write_fixtures:
        write_fixtures(stats, refs, cube, design)
    return 1 if results.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
