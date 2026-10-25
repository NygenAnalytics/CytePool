"""Check empirical Bayes moderation of variances against limma.

Compares ``cytepool.stats.empirical_bayes`` with limma's ``eBayes``, ``treat``,
``squeezeVar``, ``fitFDist``, ``fitFDistRobustly`` and ``fitFDistUnequalDF1`` on data
simulated from fixed seeds, with limma's ``legacy`` argument TRUE (the method of
moments, ``prior_estimator="moments"``), FALSE (the likelihood estimator,
``"likelihood"``) or NULL (limma's default, ``prior_estimator=None``). Linear model
cases (2,000 genes, 6 to 40 samples) are fitted by ``lmFit`` in R and by
``fit_linear_model`` here, then moderated with a constant prior, an intensity trend,
robustly, and with both; they cover several contrasts (some linearly dependent),
hypervariable genes, an infinite prior df, missing values, observation weights (some
of them zero), an all-zero contrast, and 1 to 25 genes, which with one or two genes
are not moderated (some of them all zero). Residual df differ between genes
with missing values and zero weights, where limma's default is the likelihood
estimator; the moment estimator is checked there with ``legacy = TRUE``, and the
likelihood estimator on equal df with ``legacy = FALSE``. Variance cases call the
hyperparameter estimators directly: 2 to 6,000 variances with and without a
covariate, few distinct or infinite covariate values, zero, missing and unusable
variances, fractional df, prior weights, planted outliers with equal and unequal df,
infinite robust prior df, three to five trended variances (limma fits a straight
line through at most four), two variances above zero among zeros, and covariates on
scales of 1e-5 and 1e-9. The weighted LOWESS of the likelihood estimator's trend is
also checked against limma's ``weightedLowess``: ties, windows narrower than 1e-7,
windows whose x barely vary, two distant clusters, and the trend's weights.

Each row of the table covers one case and option set: the largest difference over all
outputs compared (eBayes: t, p, B, F and its p-value, the priors, posterior variances,
total df, var.prior; treat: t and p; squeezeVar and the fitFDist functions: every
component limma returns). Differences are relative, except for t, B, F and outlier
probabilities, scaled by max(|reference|, 1), and p-values, scaled by
max(p, 1e-300) max(1, |log p|), about the error in log p relative to |log p|: an
error in the prior df moves a p-value in the far tail in proportion to |log p|, and
doubles below 1e-300 carry no relative precision.

Tiers: closed forms and the moment estimator are "exact" (1e-10). The robust moment
estimator is "close" (1e-6): limma solves its Winsorized moment equation only to a
tolerance, which leaves its prior df within about 1e-7 of the exact root. The
likelihood estimator is "close" too: its prior df is where R's ``optimize`` stops,
which both programs reach by the same steps, up to rounding. The exact moderated
F-statistic of a fit with missing values or weights has no limma counterpart, but it
equals limma's F of the same data coded as group means, whose coefficients are
uncorrelated for every gene; it is checked against that.

Deliberate differences, printed as such: the robust likelihood estimator leaves out
features without a usable variance, which limma's counts in its test for outliers
(a missing variance as a zero one), and keeps prior weights in its second fit, which
limma's drops. Controls check the first against limma on the usable features alone,
with ``fitFDistUnequalDF1`` and with ``eBayes`` (constant prior).

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor package limma (3.61.8 or later). Set
    ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/empirical_bayes.py
    PYTHONPATH=. python validation/empirical_bayes.py --write-fixtures
"""

import argparse
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from cytepool.stats.empirical_bayes import (
    fit_f_distribution,
    fit_f_distribution_likelihood,
    fit_f_distribution_robust,
    moderated_statistics,
    squeeze_variances,
    treat,
)
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.smoothing import _weighted_lowess

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "empirical_bayes"
)
EXACT = 1e-10
CLOSE = 1e-6
LFC = float(np.log2(1.2))
# limma's legacy argument and the matching prior_estimator.
ESTIMATORS = {"TRUE": "moments", "FALSE": "likelihood", "NULL": None}
# Option sets (name: trend, robust, legacy) of the linear model cases.
OPTIONS = {
    "constant": (0, 0),
    "trend": (1, 0),
    "robust": (0, 1),
    "trend_robust": (1, 1),
}
RUNS = {name: (*options, "TRUE") for name, options in OPTIONS.items()}
for suffix, legacy in (("default", "NULL"), ("likelihood", "FALSE")):
    RUNS |= {f"{name}_{suffix}": (*opt, legacy) for name, opt in OPTIONS.items()}
# Option sets (name: robust, legacy, Winsorized tails) of the variance cases.
VARIANCE_RUNS = {
    "moments": (0, "TRUE", 0.05, 0.1),
    "robust": (1, "TRUE", 0.05, 0.1),
    "robust_wide": (1, "TRUE", 0.1, 0.2),
    "likelihood": (0, "FALSE", 0.05, 0.1),
    "likelihood_robust": (1, "FALSE", 0.05, 0.1),
}
STATISTICS = "t p.value lods F F.p.value s2.prior df.prior s2.post df.total var.prior"
ROBUST_PARTS = "scale df2 df2.shrunk tail.p.value prob.outlier df2.outlier"
LIKELIHOOD_PARTS = ["scale", "df2", "df2.shrunk", "df2.outlier"]
# Outputs whose differences are scaled by max(|reference|, floor) rather than relative.
FLOORS = dict.fromkeys(["t", "lods", "F", "treat.t", "means.F", "prob.outlier"], 1.0)
# p-values, whose scale is also multiplied by max(1, |log p|)
PROBABILITIES = ["p.value", "F.p.value", "treat.p", "means.F.p.value", "tail.p.value"]
FLOORS |= dict.fromkeys(PROBABILITIES, 1e-300)

R_SCRIPT = r"""
suppressPackageStartupMessages(library(limma))
root <- commandArgs(trailingOnly = TRUE)[[1]]
has <- function(dir, name) file.exists(file.path(dir, paste0(name, ".bin")))
rd <- function(dir, name) {
  dims <- scan(file.path(dir, paste0(name, ".dim")), quiet = TRUE)
  v <- readBin(file.path(dir, paste0(name, ".bin")), "double", n = prod(dims), size = 8)
  if (length(dims) == 2) matrix(v, dims[1], dims[2]) else v
}
wr <- function(dir, name, x) writeBin(as.double(x), file.path(dir, paste0("out_", name, ".bin")), size = 8)
legacy_of <- function(x) switch(x, "TRUE" = TRUE, "FALSE" = FALSE, "NULL" = NULL)
# Without outliers fitFDistUnequalDF1 omits df2.shrunk and df2.outlier: df2 and NA.
likelihood_parts <- function(dir, tag, f) {
  wr(dir, paste(tag, "scale"), f$scale); wr(dir, paste(tag, "df2"), f$df2)
  wr(dir, paste(tag, "df2.shrunk"), if (is.null(f$df2.shrunk)) f$df2 else f$df2.shrunk)
  wr(dir, paste(tag, "df2.outlier"), if (is.null(f$df2.outlier)) NA else f$df2.outlier)
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  runs <- read.table(file.path(dir, "runs.txt"), header = TRUE, stringsAsFactors = FALSE,
                     colClasses = c(legacy = "character"))
  if (has(dir, "y")) {
    y <- rd(dir, "y")
    w <- if (has(dir, "w")) rd(dir, "w") else NULL
    fits <- list(fit = suppressWarnings(lmFit(y, rd(dir, "X"), weights = w)))
    if (has(dir, "C")) fits$fit <- contrasts.fit(fits$fit, rd(dir, "C"))
    if (has(dir, "Xm")) fits$means <- suppressWarnings(lmFit(y, rd(dir, "Xm"), weights = w))
    for (i in seq_len(nrow(runs))) {
      tag <- runs$name[i]; tr <- runs$trend[i] == 1; rb <- runs$robust[i] == 1
      lg <- legacy_of(runs$legacy[i])
      eb <- suppressWarnings(eBayes(fits$fit, trend = tr, robust = rb, legacy = lg))
      for (k in strsplit("STATISTICS", " ")[[1]]) wr(dir, paste(tag, k), eb[[k]])
      if (has(dir, "keep") && rb && !tr) {  # the genes with residual df alone
        u <- suppressWarnings(eBayes(fits$fit[rd(dir, "keep") == 1, ], robust = TRUE, legacy = lg))
        for (k in c("s2.prior", "df.prior", "s2.post", "df.total", "t", "p.value")) wr(dir, paste(tag, "usable", k), u[[k]])
      }
      tt <- suppressWarnings(treat(fits$fit, lfc = LFC, trend = tr, robust = rb, legacy = lg))
      wr(dir, paste(tag, "treat.t"), tt$t); wr(dir, paste(tag, "treat.p"), tt$p.value)
      if (!is.null(fits$means)) {
        em <- suppressWarnings(eBayes(fits$means, trend = tr, robust = rb, legacy = lg))
        wr(dir, paste(tag, "means.F"), em$F); wr(dir, paste(tag, "means.F.p.value"), em$F.p.value)
      }
    }
  } else if (has(dir, "lowess_x")) {
    wl <- weightedLowess(rd(dir, "lowess_x"), rd(dir, "lowess_y"), rd(dir, "lowess_w"), span = rd(dir, "span"), iterations = 1)
    wr(dir, "weighted_lowess fitted", wl$fitted)
  } else {
    v <- rd(dir, "v"); d <- rd(dir, "d")
    cv <- if (has(dir, "covariate")) rd(dir, "covariate") else NULL
    pw <- if (has(dir, "pw")) rd(dir, "pw") else NULL
    for (i in seq_len(nrow(runs))) {
      tag <- runs$name[i]; rb <- runs$robust[i] == 1; lg <- legacy_of(runs$legacy[i])
      tails <- c(runs$p_lower[i], runs$p_upper[i])
      s <- suppressWarnings(squeezeVar(v, d, covariate = cv, robust = rb, winsor.tail.p = tails, legacy = lg))
      wr(dir, paste(tag, "var.post"), s$var.post); wr(dir, paste(tag, "var.prior"), s$var.prior)
      wr(dir, paste(tag, "df.prior"), s$df.prior)
      if (!isTRUE(lg)) {
        f <- suppressWarnings(fitFDistUnequalDF1(v, d, covariate = cv, robust = rb, prior.weights = pw))
        likelihood_parts(dir, tag, f)
        if (has(dir, "usable")) {
          keep <- !is.na(v) & d >= 0.01
          u <- suppressWarnings(fitFDistUnequalDF1(v[keep], d[keep], robust = rb))
          likelihood_parts(dir, paste(tag, "usable"), u)
        }
      } else if (rb) {
        r <- fitFDistRobustly(v, d, covariate = cv, winsor.tail.p = tails)
        for (k in strsplit("ROBUST_PARTS", " ")[[1]]) wr(dir, paste(tag, k), r[[k]])
      } else {
        f <- suppressWarnings(fitFDist(v, d, covariate = cv))
        wr(dir, paste(tag, "scale"), f$scale); wr(dir, paste(tag, "df2"), f$df2)
      }
    }
  }
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma"))), file.path(root, "versions.txt"))
"""
R_SCRIPT = R_SCRIPT.replace("STATISTICS", STATISTICS)
R_SCRIPT = R_SCRIPT.replace("ROBUST_PARTS", ROBUST_PARTS)
R_SCRIPT = R_SCRIPT.replace("LFC", repr(LFC))


def simulate_expression(rng, design, n_genes, n_de=0.075, n_hyper=0.02):
    """Genes x samples log-expression with a mean-variance trend and sparse effects."""
    mean = rng.uniform(2.0, 12.0, n_genes)
    sd = np.sqrt(0.3 / rng.chisquare(6.0, n_genes)) * np.exp(-0.12 * (mean - 7))
    effects = np.zeros((n_genes, design.shape[1]))
    effects[:, 0] = mean
    active = rng.random(n_genes) < n_de
    effects[active, 1:] = rng.normal(0.0, 1.5, (active.sum(), design.shape[1] - 1))
    noise = rng.normal(0.0, 1.0, (n_genes, design.shape[0]))
    y = effects @ design.T + noise * sd[:, None]
    hyper = rng.random(n_genes) < n_hyper
    y[hyper] += rng.normal(0.0, 1.5, (hyper.sum(), design.shape[0]))
    return y


def exact_variances(rng, design, variances):
    """Genes x samples values whose residual variances on the design are given."""
    n, p = design.shape
    residual_basis = np.linalg.qr(design, mode="complete")[0][:, p:]
    z = rng.normal(0.0, 1.0, (variances.size, n - p))
    z *= np.sqrt(variances * (n - p) / np.sum(z**2, axis=1))[:, None]
    effects = np.where(
        rng.random(variances.size) < 0.1, rng.normal(0, 1, z.shape[0]), 0
    )
    mean = rng.uniform(2.0, 12.0, variances.size)
    return mean[:, None] + effects[:, None] * design[:, 1] + z @ residual_basis.T


def groups_design(groups, *covariates):
    indicators = [(groups == g).astype(float) for g in np.unique(groups)[1:]]
    return np.column_stack([np.ones(groups.size), *indicators, *covariates])


def means_design(groups):
    return np.column_stack([(groups == g).astype(float) for g in np.unique(groups)])


def model_cases(rng, n):
    """Linear model cases with n genes (fewer for the few-gene cases)."""
    g2, g3, g4 = np.repeat([0, 1], 3), np.repeat([0, 1, 2], 4), np.repeat(range(4), 10)
    x2, x3 = groups_design(g2), groups_design(g3)
    x = groups_design(g3, rng.uniform(25.0, 70.0, 12))
    contrasts = np.array([[0, 1, 0, 0], [0, -1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 50]]).T
    cases = {"two_groups": {"X": x2, "y": simulate_expression(rng, x2, n)}}
    # The likelihood estimator forced on equal df (most option sets are below).
    cases["two_groups"]["runs"] = [
        *OPTIONS,
        "constant_likelihood",
        "trend_robust_likelihood",
    ]
    y = simulate_expression(rng, x, n)
    cases["contrasts"] = {"X": x, "C": contrasts.astype(float), "y": y}
    cases["contrasts"]["runs"] = ["constant", "robust"]
    x = groups_design(g4)
    cases["forty_samples"] = {"X": x, "y": simulate_expression(rng, x, n)}
    cases["forty_samples"]["runs"] = ["trend", "trend_robust"]
    # Residual variances that vary far less than chi-squared sampling allows give an
    # infinite prior df; one hypervariable gene gets a finite robust prior df.
    variances = 0.05 * np.exp(rng.normal(0.0, 0.1, n))
    variances[0] *= 40.0
    y = exact_variances(rng, x2, variances)
    cases["infinite_prior"] = {
        "X": x2,
        "y": y,
        "runs": [*OPTIONS, "constant_likelihood"],
    }
    y = simulate_expression(rng, x3, n)
    y[rng.random(y.shape) < 0.05] = np.nan
    y[0, [1, 2, 3, 5, 6, 7, 10, 11]] = np.nan  # four values in three groups: one df
    y[1, 4:8] = np.nan  # the second group missing: one coefficient not estimable
    cases["missing_values"] = {"X": x3, "Xm": means_design(g3), "y": y}
    cases["missing_values"]["runs"] = [*OPTIONS, "trend_robust_default"]
    y = simulate_expression(rng, x3, n)
    w = np.exp(rng.normal(0.0, 0.5, y.shape)) / (0.2 + 4.0 / (1.0 + np.exp(y - 5.0)))
    cases["observation_weights"] = {"X": x3, "Xm": means_design(g3), "y": y, "w": w}
    cases["observation_weights"]["runs"] = ["constant", "robust"]
    zero = np.array([[0, 1, 0], [0, 0, 0]], dtype=float).T
    y = simulate_expression(rng, x3, n)
    cases["zero_contrast"] = {"X": x3, "C": zero, "y": y, "runs": ["constant", "trend"]}
    for genes in (2, 3, 5, 25):
        runs = ["constant", "trend"] + (["robust"] if genes >= 10 else [])
        y = simulate_expression(rng, x2, genes)
        cases[f"{genes}_genes"] = {"X": x2, "y": y, "runs": runs}
    # One or two genes are not moderated, so an all-zero gene keeps a zero prior
    # variance: its t is 0 / 0, and limma's var.prior can be 1 / 0. (Zero is the
    # value whose residuals are exactly zero in both programs.)
    varied = simulate_expression(rng, x2, 1)
    for name, y in (
        ("1_gene_without_variation", np.zeros((1, 6))),
        ("2_genes_one_without_variation", np.vstack([np.zeros(6), varied])),
        ("2_genes_without_variation", np.zeros((2, 6))),
    ):
        cases[name] = {"X": x2, "y": y, "runs": list(OPTIONS)}
    return cases


def unequal_df_model_cases(rng, n):
    """Linear model cases whose residual df differ between genes."""
    default = [f"{run}_default" for run in OPTIONS]
    g3 = np.repeat([0, 1, 2], 4)
    x3 = groups_design(g3)
    y = simulate_expression(rng, x3, n)
    y[rng.random(y.shape) < 0.08] = np.nan
    cases = {"missing_values_all_df": {"X": x3, "Xm": means_design(g3), "y": y}}
    cases["missing_values_all_df"]["runs"] = default
    y = simulate_expression(rng, x3, n)
    w = np.exp(rng.normal(0.0, 0.5, y.shape))
    w[rng.random(y.shape) < 0.06] = 0.0
    cases["zero_weights"] = {"X": x3, "y": y, "w": w, "runs": default}
    # One value per group: no residual df and no variance for one gene in 50. The
    # robust fits leave those genes out, which limma's enter as zero variances; a
    # control compares the constant one with limma's of the other genes alone.
    full = simulate_expression(rng, x3, n)
    y = np.where(rng.random(full.shape) < 0.04, np.nan, full)
    y[: n // 50] = np.nan
    y[: n // 50, [0, 5, 10]] = full[: n // 50, [0, 5, 10]]
    keep = (np.arange(n) >= n // 50).astype(float)
    cases["no_residual_df"] = {"X": x3, "y": y, "keep": keep, "runs": default}
    cases["no_residual_df"]["differs"] = default[2:]
    return cases


def variance_cases(rng, n):
    """Cases of variances, degrees of freedom and covariates (at most n variances)."""
    cases = {}

    def trended(size, df=4.0, d0=5.0):
        z = rng.uniform(0.0, 10.0, size)
        return (0.1 + 0.05 * z) * rng.f(df, d0, size), z

    def case(v, d=4.0, **extra):
        return {"v": v, "d": np.atleast_1d(np.asarray(d, dtype=float)), **extra}

    for size in sorted({2, 3, 5, 6, 12, 29, 30, 200, n}):
        v, z = trended(size)
        cases[f"sizes_{size}"] = case(v, covariate=z, likelihood=["likelihood"])
        cases[f"sizes_{size}_constant"] = case(v)
    for levels in (2, 3):
        z = np.tile(np.arange(levels, dtype=float), 300 // levels + 1)[:300]
        cases[f"{levels}_covariate_levels"] = case(trended(300)[0], covariate=z)
    v, z = trended(300)
    z[np.argsort(z)[[0, -1, -2]]] = [-np.inf, np.inf, np.inf]
    cases["infinite_covariate"] = case(v, covariate=z)
    v, z = trended(300)
    v[:5] = 0.0
    cases["zero_variances"] = case(v, covariate=z, likelihood=["likelihood"])
    v, z = trended(50)
    v[:30] = 0.0
    cases["mostly_zero"] = case(v, likelihood=["likelihood", "likelihood_robust"])
    v, z = trended(300)
    d = rng.choice([2.0, 3.0, 4.0], 300, p=[0.1, 0.2, 0.7])
    d[np.argsort(z)[:2]] = 0.0  # the smallest covariates: predicted beyond the range
    v[[np.argsort(z)[-1], 7]] = np.nan
    cases["unusable"] = case(v, d, covariate=z, likelihood=["likelihood"])
    size = max(n, 300)
    for name, d in (
        ("outliers", np.full(size, 4.0)),
        ("outliers_unequal_df", rng.choice([2.0, 3.0, 4.0], size, p=[0.1, 0.2, 0.7])),
    ):
        z = rng.normal(6.0, 2.0, size)
        v = np.exp(-1.5 + 0.4 * np.sin(z)) * rng.f(d, 8.0)
        v[: size // 25] *= 12.0
        v[size // 25] = 0.0
        runs = ["likelihood_robust"]
        cases[name] = case(v, d, covariate=z, robust=True, likelihood=runs)
        cases[f"{name}_constant"] = case(v, d, robust=True, likelihood=runs)
    v = 0.3 * np.random.default_rng(104).chisquare(4.0, 300) / 4.0
    v[:3] *= [30.0, 8.0, 5.0]  # an infinite robust df2, with outliers
    cases["robust_infinite_prior"] = case(v, robust=True)
    v, z = trended(12)
    v[3] = np.nan
    cases["robust_missing"] = case(v, covariate=z, robust=True)
    cases["robust_nine"] = case(trended(9)[0], robust=True)
    # log variances spread uniformly: the moment estimate of df2 is infinite, the
    # Winsorized one is not
    cases["robust_moments_infinite"] = case(
        0.3 * np.exp(rng.uniform(-1.25, 1.25, 300)), robust=True
    )
    # Variances far less spread than chi-squared sampling gives, with two outliers:
    # an infinite robust df2, with unequal df and with a covariate.
    v = 0.3 * np.exp(rng.normal(0.0, 0.05, (2, 300)))
    v[:, :2] *= [40.0, 8.0]
    d = rng.choice([2.0, 3.0, 6.0], 300)
    cases["robust_infinite_unequal_df"] = case(v[0], d, robust=True)
    z = rng.uniform(0.0, 10.0, 300)
    cases["robust_infinite_trend"] = case(v[1], covariate=z, robust=True)
    return cases


def likelihood_variance_cases(rng, n):
    """Variance cases for the likelihood estimator alone (at most n variances)."""
    both = ["likelihood", "likelihood_robust"]

    def sample(size, df, d0=8.0):
        z = rng.uniform(0.0, 10.0, size)
        return np.exp(-1.0 + 0.15 * z) * rng.f(df, d0), z

    cases = {}
    d = rng.uniform(0.05, 3.0, n)
    v, z = sample(n, d)
    v[: n // 50] *= 30.0
    cases["fractional_df"] = {"v": v, "d": d, "covariate": z, "likelihood": both}
    d = rng.choice([1.0, 2.0, 5.0, 12.0], n)
    v, z = sample(n, d)
    v[: n // 40] *= 20.0
    weights = rng.uniform(0.2, 2.0, n)
    weights[n // 40 : n // 40 + n // 30] = 0.0
    # The robust fit keeps the prior weights in its second fit; limma's drops them.
    cases["prior_weights"] = {"v": v, "d": d, "covariate": z, "pw": weights}
    cases["prior_weights"] |= {"likelihood": both, "differs": ["likelihood_robust"]}
    d = rng.choice([2.0, 4.0, 6.0], n)
    v, _ = sample(n, d)
    v[: n // 40] *= 15.0
    v[n // 40 : n // 40 + 8] = np.nan
    d[n // 40 + 8 : n // 40 + 14] = [0.0, 0.0, 0.004, 0.004, 0.0, 0.0]
    # Without a covariate, CytePool's robust fit equals limma's of the usable ones.
    cases["robust_unusable"] = {"v": v, "d": d, "usable": np.ones(1)}
    cases["robust_unusable"] |= {"likelihood": both, "differs": ["likelihood_robust"]}
    size = 3 * n  # above 500: a span below 1, and anchors 200 at most apart
    d = rng.choice([2.0, 3.0, 4.0, 8.0], size)
    v, z = sample(size, d)
    v[: size // 60] *= 25.0
    v[size // 60 : size // 30] *= 0.02  # small variances: lower-tail outliers
    cases["large_trend"] = {"v": v, "d": d, "covariate": z, "likelihood": both}
    # Variances that vary less than chi-squared sampling explains: d0 near its
    # largest value; and variances that vary more than any d0 above 2 explains.
    d = rng.choice([3.0, 4.0], 300)
    v = 0.4 * np.exp(rng.normal(0.0, 0.05, 300))
    cases["flat_likelihood"] = {"v": v, "d": d, "likelihood": ["likelihood"]}
    d = rng.choice([3.0, 4.0], 300)
    v = 0.4 * rng.f(d, 0.6)
    cases["heavy_tails"] = {"v": v, "d": d, "likelihood": both}
    # limma's loessFit fits a weighted line through at most four features (a missing
    # variance counts) and a weighted LOWESS through five.
    for size in (3, 4, 5):
        d = rng.choice([2.0, 3.0, 5.0, 8.0], size)
        v, z = sample(size, d)
        cases[f"few_trended_{size}"] = {"v": v, "d": d, "covariate": z}
    v = cases["few_trended_4"]["v"].copy()
    v[3] = np.nan  # not robust: limma would count it in its test for outliers
    cases["few_trended_missing"] = cases["few_trended_4"] | {"v": v}
    cases["few_trended_missing"]["likelihood"] = ["likelihood"]
    # Two variances above zero: a constant location and no robust stage.
    d = rng.choice([2.0, 4.0], 30)
    v, z = sample(30, d)
    v[2:] = 0.0
    cases["two_positive"] = {"v": v, "d": d, "covariate": z}
    # Covariates on small scales: the weighted LOWESS fits means in windows whose
    # weighted sum of squares of x is below 1e-7, and prior-weighted means in
    # windows narrower than 1e-7, tied values included.
    d = rng.choice([2.0, 3.0, 5.0, 8.0], n)
    v, z = sample(n, d)
    cases["flat_covariate"] = {"v": v, "d": d, "covariate": 1e-5 * z}
    for case in cases.values():
        case.setdefault("likelihood", both)
    d = rng.choice([2.0, 3.0, 5.0, 8.0], n)
    v, z = sample(n, d)
    z = 1e-9 * np.round(z, 1)
    cases["narrow_covariate"] = {"v": v, "d": d, "covariate": z, "likelihood": both}
    return {name: case | {"likelihood_only": True} for name, case in cases.items()}


def weighted_lowess_cases(rng):
    """Inputs of limma's weighted LOWESS, which the likelihood estimator's trend uses."""
    narrow = np.cumsum(rng.integers(0, 3, 60)) * 2.0**-26  # windows below 1e-7
    clusters = np.concatenate((np.zeros(50), np.full(50, 1e3)))
    inputs = {
        "ties": (np.round(rng.uniform(0.0, 5.0, 150), 1), 0.3),
        "narrow_windows": (narrow, 0.3),
        "flat_windows": (1e-5 * rng.uniform(0.0, 1.0, 80), 0.5),
        "two_clusters": (clusters + rng.uniform(0.0, 1e-4, 100), 0.3),
        "trend_weights": (rng.uniform(0.0, 8.0, 700), 0.9),
    }
    cases = {}
    for name, (x, span) in inputs.items():
        w = rng.uniform(0.2, 3.0, x.size)
        if name == "trend_weights":  # as the trend weights unusable variances
            w[rng.random(x.size) < 0.1] = 1e-8
        y = rng.normal(0.0, 1.0, x.size)
        cases[f"weighted_lowess_{name}"] = {
            "lowess_x": x, "lowess_y": y, "lowess_w": w, "span": np.array([span]),
        }  # fmt: skip
    return cases


def write_cases(cases, root):
    for name, case in cases.items():
        directory = root / name
        directory.mkdir()
        for key, value in case.items():
            if isinstance(value, np.ndarray):
                (directory / f"{key}.bin").write_bytes(value.tobytes(order="F"))
                (directory / f"{key}.dim").write_text(" ".join(map(str, value.shape)))
        lines = ["name trend robust legacy p_lower p_upper"]
        for run in case_runs(case):
            if "y" in case:
                trend, robust, legacy = RUNS[run]
                lines.append(f"{run} {trend} {robust} {legacy} 0.05 0.1")
            else:
                robust, legacy, lower, upper = VARIANCE_RUNS[run]
                lines.append(f"{run} 0 {robust} {legacy} {lower} {upper}")
        (directory / "runs.txt").write_text("\n".join(lines) + "\n")


def case_runs(case):
    """The option sets of a case."""
    if "lowess_x" in case:
        return []
    if "y" in case:
        return case["runs"]
    runs = []
    if not case.get("likelihood_only"):
        runs = ["robust", "robust_wide"] if case.get("robust") else ["moments"]
    return runs + case.get("likelihood", [])


def run_limma(cases):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_cases(cases, root)
        (root / "cases.txt").write_text("\n".join(cases) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        for name, case in cases.items():
            case["limma"] = {}
            for path in (root / name).glob("out_*.bin"):
                values = np.fromfile(path, dtype="<f8")
                case["limma"][path.stem[4:]] = np.where(
                    np.isnan(values), np.nan, values
                )
        return (root / "versions.txt").read_text().split("\n")[:2]


def difference(ours, reference, floor=0.0, log_scaled=False):
    """Largest |ours - reference| / scale; inf if NaN or infinities differ.

    The scale is max(|reference|, floor), times max(1, |log scale|) if log_scaled.
    """
    ours, reference = np.broadcast_arrays(
        np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    )
    nan = np.isnan(reference)
    infinite = np.isinf(reference)
    if not np.array_equal(np.isnan(ours), nan) or np.any(
        ours[infinite] != reference[infinite]
    ):
        return np.inf
    finite = ~(nan | infinite)
    if not finite.any():
        return 0.0
    scale = np.maximum(np.abs(reference[finite]), floor)
    scale = np.where(scale > 0, scale, 1.0)
    if log_scaled:
        scale *= np.maximum(1.0, np.abs(np.log(scale)))
    return float(np.max(np.abs(ours[finite] - reference[finite]) / scale))


def model_results(case, run):
    """CytePool's statistics of a linear model case, keyed by limma's names."""
    fit = fit_linear_model(
        case["y"].T, case["X"], weights=None if "w" not in case else case["w"].T
    )
    if "C" in case:
        fit = contrasts_fit(fit, case["C"], method="limma")
    trend, robust, legacy = RUNS[run]
    options = {
        "trend": bool(trend),
        "robust": bool(robust),
        "prior_estimator": ESTIMATORS[legacy],
    }
    eb = moderated_statistics(fit, f_statistic="limma", **options)
    tr = treat(fit, lfc=LFC, **options)
    out = {
        "t": eb.t, "p.value": eb.p_value, "lods": eb.lods, "F": eb.f_statistic,
        "F.p.value": eb.f_p_value, "s2.prior": eb.prior_variance, "df.prior": eb.prior_df,
        "s2.post": eb.posterior_variance, "df.total": eb.df_total,
        "var.prior": eb.coefficient_prior_variance, "treat.p": tr.p_value,
        # limma reports t = 0 for coefficients that are NaN; here they stay NaN
        "treat.t": np.where(np.isnan(fit.coefficients), 0.0, tr.t),
    }  # fmt: skip
    if "Xm" in case:
        exact = moderated_statistics(fit, f_statistic="exact", **options)
        out["means.F"], out["means.F.p.value"] = exact.f_statistic, exact.f_p_value
    if "keep" in case and robust and not trend:  # limma's of the other genes alone
        keep = case["keep"] == 1
        for key in ("s2.prior", "df.prior", "s2.post", "df.total", "t", "p.value"):
            value = out[key]
            out[f"usable {key}"] = value[keep] if np.ndim(value) else value
    return out


def variance_results(case, run):
    robust, legacy, lower, upper = VARIANCE_RUNS[run]
    keywords = {"covariate": case.get("covariate")}
    squeezed = squeeze_variances(
        case["v"],
        case["d"],
        robust=bool(robust),
        winsor_tail_p=(lower, upper),
        prior_estimator=ESTIMATORS[legacy],
        **keywords,
    )
    out = {
        "var.post": squeezed.posterior_variance,
        "var.prior": squeezed.prior_variance,
        "df.prior": squeezed.prior_df,
    }
    if legacy == "FALSE":
        fit = fit_f_distribution_likelihood(
            case["v"], case["d"], prior_weights=case.get("pw"), robust=bool(robust),
            **keywords,
        )  # fmt: skip
        parts = dict(zip(LIKELIHOOD_PARTS, fit, strict=True))
        out |= parts
        if "usable" in case:  # limma's fit of the usable variances alone
            keep = ~np.isnan(case["v"]) & (case["d"] >= 0.01)
            parts["df2.shrunk"] = parts["df2.shrunk"][keep]
            out |= {f"usable {key}": value for key, value in parts.items()}
    elif robust:
        fit = fit_f_distribution_robust(
            case["v"], case["d"], winsor_tail_p=(lower, upper), **keywords
        )
        out |= {
            "scale": fit.scale, "df2": fit.df2, "df2.shrunk": fit.df2_shrunk,
            "tail.p.value": fit.tail_p_value, "prob.outlier": fit.prob_outlier,
            "df2.outlier": fit.df2_outlier,
        }  # fmt: skip
    else:
        out["scale"], out["df2"] = fit_f_distribution(case["v"], case["d"], **keywords)
    return out


def compare(cases, table):
    """Rows per case and option set: the largest difference over its outputs."""
    for name, case in cases.items():
        if "lowess_x" in case:
            x, y, w = case["lowess_x"], case["lowess_y"], case["lowess_w"]
            ours = _weighted_lowess(x, y, w, float(case["span"][0]))
            value = difference(ours, case["limma"]["weighted_lowess fitted"], 1.0)
            check = f"{name}: {x.size} fitted values"
            table.append((check, "weightedLowess", value, EXACT, False))
            continue
        model = "y" in case
        for run in case_runs(case):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ours = (model_results if model else variance_results)(case, run)
            if model:
                legacy = RUNS[run][2]
                source = "eBayes, treat" + (", group means" if "Xm" in case else "")
                exact = legacy == "TRUE" and not RUNS[run][1]
            else:
                legacy = VARIANCE_RUNS[run][1]
                source = "squeezeVar, " + (
                    "fitFDistUnequalDF1"
                    if legacy == "FALSE"
                    else "fitFDistRobustly"
                    if VARIANCE_RUNS[run][0]
                    else "fitFDist"
                )
                exact = run == "moments"
            source += f" (legacy = {legacy})"
            tolerance = EXACT if exact else CLOSE
            differs = run in case.get("differs", ())
            rows = {"": {}, "usable ": {}}
            for key, value in ours.items():
                reference = case["limma"].get(f"{run} {key}")
                if reference is None or reference.size == 0:
                    continue  # limma does not report it in this case
                if np.ndim(value) == 2:  # R writes matrices column by column
                    reference = reference.reshape(-1, np.shape(value)[0]).T
                part = "usable " if key.startswith("usable ") else ""
                base = key.removeprefix("usable ")
                floor, log_scaled = FLOORS.get(base, 0.0), base in PROBABILITIES
                rows[part][key] = difference(value, reference, floor, log_scaled)
                # One value where limma reports one (df2.shrunk is always a vector);
                # a deliberate difference may change which features are outliers.
                single = np.size(value) == 1
                shaped = base == "df2.shrunk" or single == (reference.size == 1)
                if not (shaped or differs and not part):
                    rows[part][key] = np.inf
            for part, differences in rows.items():
                if not differences:
                    continue
                worst = max(differences, key=differences.__getitem__)
                check = f"{name} {run}: {len(differences)} outputs, largest {worst}"
                if part:
                    check += ", against limma's fit of the usable features alone"
                    table.append((check, source, differences[worst], tolerance, False))
                else:
                    table.append(
                        (check, source, differences[worst], tolerance, differs)
                    )


def write_fixtures(cases):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, case in cases.items():
        arrays = {k: v for k, v in case.items() if isinstance(v, np.ndarray)}
        arrays |= {f"limma {k}": v for k, v in case["limma"].items()}
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = model_cases(np.random.default_rng(1), 2000) | variance_cases(
        np.random.default_rng(2), 2000
    )
    cases |= unequal_df_model_cases(np.random.default_rng(5), 2000)
    cases |= likelihood_variance_cases(np.random.default_rng(6), 2000)
    cases |= weighted_lowess_cases(np.random.default_rng(9))
    small = model_cases(np.random.default_rng(3), 100) | variance_cases(
        np.random.default_rng(4), 200
    )
    small |= unequal_df_model_cases(np.random.default_rng(7), 100)
    small |= likelihood_variance_cases(np.random.default_rng(8), 200)
    small |= weighted_lowess_cases(np.random.default_rng(10))
    small = {f"fixture_{name}": case for name, case in small.items()}
    versions = run_limma(cases | small if arguments.write_fixtures else cases)
    print("Reference: " + ", ".join(versions))
    table = []
    compare(cases | small if arguments.write_fixtures else cases, table)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    failures = 0
    for check, source, value, tolerance, differs in table:
        if differs:
            result, bound = "deliberate difference", "n/a"
        else:
            result, bound = (
                ("pass" if value <= tolerance else "FAIL"),
                f"{tolerance:.0e}",
            )
        failures += result == "FAIL"
        print(
            f"| {check} | {source} | max scaled difference | {value:.1e} "
            f"| {bound} | {result} |"
        )
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(
                {name.removeprefix("fixture_"): case for name, case in small.items()}
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
