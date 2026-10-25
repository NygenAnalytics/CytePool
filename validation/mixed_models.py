"""Check linear mixed models and their Satterthwaite tests against lme4 and lmerTest.

Compares ``cytepool.stats.mixed_models.fit_mixed_model`` with lme4's ``lmer`` (REML)
and ``fixed_effect_tests`` with lmerTest's Satterthwaite t-tests (``summary`` for every
coefficient, ``contest1D`` for a contrast), gene by gene, on data simulated from fixed
seeds: one random intercept; unbalanced groups, including donors with a single sample;
two crossed and two nested random intercepts; a design simulated without donor
variance, so that most fits lie on the boundary; observation weights like voom's, with
one and with two random intercepts; random intercepts up to 30 times the residual
standard deviation; and 2,000 genes. The variance of each random intercept is drawn
per gene, a fifth of them zero.

Two questions are kept apart. The optimum: lme4 fits each gene from its own start,
with the tolerances of its default optimiser (BOBYQA from nloptr) tightened to
``xtol_abs = xtol_rel = 1e-14`` and ``ftol_abs = ftol_rel = 1e-16`` so that it
converges (with ``xtol_abs = 1e-12`` and ``ftol_abs = 1e-14`` alone, some fits stopped
with a criterion 3e-7 above the minimum); its REML criterion (``REMLcrit``), which
CytePool's may not exceed, is compared to the exact tier, and the fixed effects, their
standard errors, the variance components and the residual variance to the close tier
(1e-5 where the variance ratios are large; see ``LARGE_RATIOS``).
The computations at given estimates: lme4 and lmerTest evaluated once at CytePool's
estimates (passed as the starting values, with one evaluation allowed) give the REML
criterion, the fixed effects, their standard errors and the t-statistics, closed forms
compared to the exact tier, and Satterthwaite's degrees of freedom and the p-values,
compared to 1e-5 because lmerTest differentiates numerically in theta (see
``SATTERTHWAITE``). For information, the script reports the degrees of freedom at
lme4's own optimum, which inherit the optimiser's precision where a variance is small,
and the variance components and degrees of freedom with lme4's default tolerances.

With ``--benchmark``, the script times ``fit_mixed_model`` on one thread on 20,000
genes x 120 samples with observation weights, for one, two crossed and two nested
random intercepts, and lmerTest's ``lmer`` and ``summary`` on 100 of the genes.

Deliberate differences, printed as such: CytePool refuses a zero weight (lme4's REML
criterion is then infinite) and a missing value (lme4 drops the sample); it leaves
unfitted values constant within every donor, which the fixed and random effects
reproduce (lme4 warns, with a residual standard deviation near zero); and it refuses
a design whose fixed and random effects span every sample, which lme4 fits.

Requirements:
    Python 3.13 with numpy, numba and mpmath, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the CRAN packages lme4, lmerTest and jsonlite. Set ``RSCRIPT``
    to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/mixed_models.py
    PYTHONPATH=. python validation/mixed_models.py --write-fixtures
    PYTHONPATH=. python validation/mixed_models.py --benchmark
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.design import Design
from cytepool.stats.mixed_models import fit_mixed_model, fixed_effect_tests

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "mixed_models"
)
EXACT = 1e-10
CLOSE = 1e-6
# lmerTest differentiates the deviance numerically in theta, on which it depends
# through theta^2 only: at an interior optimum its curvature in theta is about
# 4 theta^2 times that in theta^2, so the numerical Hessian loses relative precision
# like 1 / theta^2 for a small variance. Its degrees of freedom, and the p-values that
# use them, are compared to 1e-5; CytePool's derivatives are exact.
SATTERTHWAITE = 1e-5
# Where theta is large, the criterion is nearly flat in it, and lme4's optimiser,
# which stops on changes in the criterion, places theta to about 1e-6 only (the
# gene that differs most is checked against a 30-digit optimum).
LARGE_RATIOS = 1e-5
N_GENES = 100
N_MANY = 2000
N_FIXTURE_GENES = 12

R_SCRIPT = r"""
suppressPackageStartupMessages({library(lme4); library(lmerTest)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_doubles <- function(dir, name, n) {
  readBin(file.path(dir, name), what = "double", n = n, size = 8, endian = "little")
}
write_doubles <- function(x, dir, name) {
  writeBin(as.double(x), file.path(dir, name), size = 8, endian = "little")
}
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
# lme4's default optimiser, BOBYQA from nloptr, with tightened tolerances.
tight <- lmerControl(
  optimizer = "nloptwrap", calc.derivs = FALSE, check.conv.singular = "ignore",
  optCtrl = list(
    xtol_abs = 1e-14, ftol_abs = 1e-16, xtol_rel = 1e-14, ftol_rel = 1e-16, maxeval = 1e5
  )
)
# A single evaluation at the starting values: lme4 and lmerTest at given estimates.
given <- lmerControl(
  optimizer = "nloptwrap", calc.derivs = FALSE, check.conv.singular = "ignore",
  restart_edge = FALSE, boundary.tol = 0, optCtrl = list(maxeval = 1)
)
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  meta <- jsonlite::fromJSON(file.path(dir, "meta.json"))
  n <- meta$n; p <- meta$p; G <- meta$G; K <- length(meta$components)
  d <- as.data.frame(meta$groups, stringsAsFactors = FALSE)
  d$X <- matrix(read_doubles(dir, "x.bin", n * p), n, p)
  y <- matrix(read_doubles(dir, "y.bin", G * n), G, n, byrow = TRUE)
  w <- if (meta$weights) matrix(read_doubles(dir, "w.bin", G * n), G, n, byrow = TRUE) else NULL
  ours <- matrix(read_doubles(dir, "theta.bin", G * K), G, K, byrow = TRUE)
  L <- read_doubles(dir, "contrast.bin", p)
  form <- as.formula(paste("y ~ 0 + X +", meta$random))
  rows <- function(k) matrix(NA_real_, G, k)
  out <- list(
    coefficients = rows(p), se = rows(p), components = rows(K), theta = rows(K),
    sigma2 = rep(NA_real_, G), criterion = rep(NA_real_, G), df_at_optimum = rows(p),
    given_coefficients = rows(p), given_se = rows(p), given_criterion = rep(NA_real_, G),
    df = rows(p), t = rows(p), p = rows(p), contrast = rows(5),
    default_components = rows(K), default_df = rows(p)
  )
  components_of <- function(fit) {
    vc <- as.data.frame(VarCorr(fit))
    vc$vcov[match(meta$components, vc$grp)]
  }
  tested <- function(fit) inherits(fit, "lmerModLmerTest")
  for (g in seq_len(G)) {
    d$y <- y[g, ]
    d$w <- if (is.null(w)) rep(1, n) else w[g, ]
    # The optimum, from lme4's own start.
    fit <- quiet(lmerTest::lmer(form, data = d, weights = w, REML = TRUE, control = tight))
    out$coefficients[g, ] <- fixef(fit)
    out$se[g, ] <- sqrt(diag(as.matrix(vcov(fit))))
    out$components[g, ] <- components_of(fit)
    theta <- getME(fit, "theta")
    groups <- sub("\\.\\(Intercept\\)$", "", names(theta))
    out$theta[g, ] <- theta[match(meta$components, groups)]
    out$sigma2[g] <- sigma(fit)^2
    out$criterion[g] <- REMLcrit(fit)
    if (tested(fit)) out$df_at_optimum[g, ] <- summary(fit)$coefficients[, "df"]
    # Everything at CytePool's estimates, in lme4's order of the terms.
    start <- list(theta = ours[g, match(groups, meta$components)])
    fit <- quiet(lmerTest::lmer(
      form, data = d, weights = w, REML = TRUE, start = start, control = given
    ))
    out$given_coefficients[g, ] <- fixef(fit)
    out$given_se[g, ] <- sqrt(diag(as.matrix(vcov(fit))))
    out$given_criterion[g] <- REMLcrit(fit)
    if (tested(fit)) {
      table <- summary(fit)$coefficients
      out$df[g, ] <- table[, "df"]
      out$t[g, ] <- table[, "t value"]
      out$p[g, ] <- table[, "Pr(>|t|)"]
      result <- contest1D(fit, L)
      out$contrast[g, ] <- unlist(result[1, c("Estimate", "Std. Error", "df", "t value", "Pr(>|t|)")])
    }
    if (meta$defaults) {
      fit <- quiet(lmerTest::lmer(form, data = d, weights = w, REML = TRUE))
      out$default_components[g, ] <- components_of(fit)
      if (tested(fit)) out$default_df[g, ] <- summary(fit)$coefficients[, "df"]
    }
  }
  for (name in names(out)) {
    values <- out[[name]]
    write_doubles(if (is.matrix(values)) t(values) else values, dir, paste0("lme4_", name, ".bin"))
  }
}
# Inputs where CytePool deliberately differs: what lme4 does with each.
outcomes <- character(0)
for (case in readLines(file.path(root, "deviations.txt"))) {
  meta <- jsonlite::fromJSON(file.path(root, paste0(case, ".json")))
  d <- as.data.frame(meta$samples, stringsAsFactors = FALSE)
  warned <- FALSE
  fit <- withCallingHandlers(
    suppressMessages(lme4::lmer(as.formula(meta$formula), data = d, weights = w)),
    warning = function(condition) {
      warned <<- TRUE
      invokeRestart("muffleWarning")
    }
  )
  outcomes <- c(outcomes, paste(case, REMLcrit(fit), nobs(fit), sigma(fit), warned))
}
writeLines(outcomes, file.path(root, "lme4_deviations.txt"))
writeLines(
  c(paste("R", as.character(getRversion())),
    paste("lme4", as.character(packageVersion("lme4"))),
    paste("lmerTest", as.character(packageVersion("lmerTest")))),
  file.path(root, "versions.txt")
)
"""


# lmerTest's lmer and summary, gene by gene, timed per gene.
R_BENCHMARK = r"""
suppressPackageStartupMessages(library(lmerTest))
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_doubles <- function(dir, name, n) {
  readBin(file.path(dir, name), what = "double", n = n, size = 8, endian = "little")
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  meta <- jsonlite::fromJSON(file.path(dir, "meta.json"))
  n <- meta$n; p <- meta$p; G <- meta$G
  d <- as.data.frame(meta$groups, stringsAsFactors = FALSE)
  d$X <- matrix(read_doubles(dir, "x.bin", n * p), n, p)
  y <- matrix(read_doubles(dir, "y.bin", G * n), G, n, byrow = TRUE)
  w <- matrix(read_doubles(dir, "w.bin", G * n), G, n, byrow = TRUE)
  form <- as.formula(paste("y ~ 0 + X +", meta$random))
  fit <- function(g) {
    d$y <- y[g, ]
    d$w <- w[g, ]
    summary(suppressWarnings(suppressMessages(lmer(form, data = d, weights = w))))
  }
  invisible(fit(1))
  start <- proc.time()[["elapsed"]]
  for (g in seq_len(G)) table <- fit(g)$coefficients
  seconds <- (proc.time()[["elapsed"]] - start) / G
  cat(case, seconds, "\n", file = file.path(root, "seconds.txt"), append = TRUE)
}
"""


@dataclass
class Case:
    """Inputs of one comparison. ``y`` and ``weights`` are samples x genes."""

    name: str
    samples: dict[str, Any]
    formula: str
    random: str
    y: np.ndarray = field(repr=False)
    weights: np.ndarray | None = field(default=None, repr=False)
    defaults: bool = True
    # Tolerance of the estimates against lme4's own optimum; see LARGE_RATIOS.
    optimum: float = 1e-6

    @property
    def design(self) -> Design:
        return Design(self.samples, self.formula, random=self.random)


def group_codes(design: Design) -> list[np.ndarray]:
    """The level of each sample for every grouping factor, from the sample table."""
    codes = []
    for effect in design.random:
        parts = [design.samples[name].astype(str) for name in effect.grouping]
        joined = np.array(["\t".join(values) for values in zip(*parts)])
        codes.append(np.unique(joined, return_inverse=True)[1])
    return codes


def labels(prefix: str, codes: np.ndarray) -> np.ndarray:
    """Text labels such as D01, D02, ..., which sort the same in every locale."""
    return np.array([f"{prefix}{code + 1:02d}" for code in codes], dtype=object)


def simulate(
    rng: np.random.Generator, design: Design, n_genes: int, *, random_sd: float
) -> np.ndarray:
    """Expression with gene-wise effects, residual variances and random intercepts.

    The standard deviation of each random intercept is drawn per gene, between zero
    and ``random_sd`` times the gene's residual standard deviation, and is zero for a
    fifth of the genes.
    """
    x = design.matrix
    n, p = x.shape
    effects = rng.normal(0.0, 1.0, (p, n_genes)) * (rng.random((p, n_genes)) < 0.5)
    effects[0] = rng.uniform(2.0, 10.0, n_genes)
    sd = 0.3 * np.sqrt(4.0 / rng.chisquare(4.0, n_genes))
    y = x @ effects
    for codes in group_codes(design):
        scale = rng.uniform(0.0, random_sd, n_genes) * (rng.random(n_genes) > 0.2)
        y = y + rng.normal(0.0, 1.0, (codes.max() + 1, n_genes))[codes] * scale * sd
    return y + rng.normal(0.0, 1.0, (n, n_genes)) * sd


def observation_weights(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """Positive weights that rise with expression, like voom's."""
    return np.exp(rng.normal(0.0, 0.5, y.shape)) / (0.2 + 4.0 / (1.0 + np.exp(y - 5.0)))


def with_weights(rng: np.random.Generator, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Observation weights, and y with noise of variance inverse to them added."""
    weights = observation_weights(rng, y)
    return y + rng.normal(0.0, 0.3, y.shape) / np.sqrt(weights), weights


def make_cases(n_genes: int, n_many: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases = []

    # One random intercept: 8 donors x 3 samples, a condition within donors and an
    # age per donor.
    donor = np.repeat(np.arange(8), 3)
    samples = {
        "donor": labels("D", donor),
        "condition": np.tile(np.array(["ctrl", "stim", "stim"], dtype=object), 8),
        "age": rng.uniform(20.0, 80.0, 8)[donor],
    }
    case = Case("one_intercept", samples, "~ condition + age", "(1 | donor)", None)
    case.y = simulate(rng, case.design, n_genes, random_sd=2.0)
    cases.append(case)

    # Unbalanced: 12 donors with 1 to 6 samples each.
    sizes = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    donor = np.repeat(np.arange(sizes.size), sizes)
    samples = {
        "donor": labels("D", donor),
        "condition": rng.choice(np.array(["ctrl", "stim"], dtype=object), donor.size),
        "x": rng.normal(0.0, 1.0, donor.size),
    }
    case = Case("unbalanced", samples, "~ condition + x", "(1 | donor)", None)
    case.y = simulate(rng, case.design, n_genes, random_sd=2.0)
    cases.append(case)

    # Two crossed random intercepts: 10 donors x 4 samples spread over 5 batches.
    donor = np.repeat(np.arange(10), 4)
    crossed = {
        "donor": labels("D", donor),
        "batch": labels("B", rng.integers(0, 5, donor.size)),
        "condition": np.tile(
            np.array(["ctrl", "ctrl", "stim", "stim"], dtype=object), 10
        ),
    }
    case = Case("crossed", crossed, "~ condition", "(1 | donor) + (1 | batch)", None)
    case.y = simulate(rng, case.design, n_genes, random_sd=2.0)
    cases.append(case)

    # Two nested random intercepts: 6 donors x 3 samples x 2 replicates.
    donor = np.repeat(np.arange(6), 6)
    samples = {
        "donor": labels("D", donor),
        "sample": labels("S", np.repeat(np.arange(3), 2)[np.arange(36) % 6]),
        "condition": np.tile(
            np.repeat(np.array(["ctrl", "stim", "stim"], dtype=object), 2), 6
        ),
    }
    case = Case("nested", samples, "~ condition", "(1 | donor/sample)", None)
    case.y = simulate(rng, case.design, n_genes, random_sd=2.0)
    cases.append(case)

    # A variance component at the boundary: no donor variance is simulated.
    donor = np.repeat(np.arange(8), 3)
    samples = {
        "donor": labels("D", donor),
        "condition": np.tile(np.array(["ctrl", "stim", "stim"], dtype=object), 8),
        "x": rng.normal(0.0, 1.0, donor.size),
    }
    case = Case("boundary", samples, "~ condition + x", "(1 | donor)", None)
    case.y = simulate(rng, case.design, n_genes, random_sd=0.0)
    cases.append(case)

    # Observation weights: 10 donors x 4 samples.
    donor = np.repeat(np.arange(10), 4)
    samples = {
        "donor": labels("D", donor),
        "condition": np.tile(
            np.array(["ctrl", "ctrl", "stim", "stim"], dtype=object), 10
        ),
        "x": rng.normal(0.0, 1.0, donor.size),
    }
    case = Case("observation_weights", samples, "~ condition + x", "(1 | donor)", None)
    case.y, case.weights = with_weights(
        rng, simulate(rng, case.design, n_genes, random_sd=2.0)
    )
    cases.append(case)

    # Observation weights with two crossed random intercepts.
    case = Case(
        "weights_crossed", crossed, "~ condition", "(1 | donor) + (1 | batch)", None
    )
    case.y, case.weights = with_weights(
        rng, simulate(rng, case.design, n_genes, random_sd=2.0)
    )
    cases.append(case)

    # Large variance ratios: the crossed design with standard deviations of the random
    # intercepts up to 30 times the residual one.
    case = Case(
        "large_ratios", crossed, "~ condition", "(1 | donor) + (1 | batch)", None
    )
    case.y = simulate(rng, case.design, n_genes, random_sd=30.0)
    case.optimum = LARGE_RATIOS
    cases.append(case)

    # Many genes: 16 donors (8 cases, 8 controls) x 4 samples, observation weights.
    donor = np.repeat(np.arange(16), 4)
    samples = {
        "donor": labels("D", donor),
        "disease": np.where(donor < 8, "case", "control").astype(object),
        "condition": np.tile(
            np.array(["ctrl", "ctrl", "stim", "stim"], dtype=object), 16
        ),
    }
    case = Case(
        "many_genes", samples, "~ disease + condition", "(1 | donor)", None, None, False
    )
    case.y, case.weights = with_weights(
        rng, simulate(rng, case.design, n_many, random_sd=2.0)
    )
    cases.append(case)
    return cases


@dataclass
class Deviation:
    """An input where CytePool deliberately differs from lme4."""

    samples: dict[str, Any]
    formula: str
    random: str
    y: np.ndarray
    weights: np.ndarray


def deviation_cases(case: Case, rng: np.random.Generator) -> dict[str, Deviation]:
    """The first gene of ``case`` with a zero weight, with a missing value, and made
    constant within every donor; and a design whose effects span every sample."""
    y = case.y[:, 0]
    ones = np.ones(y.size)
    zero = ones.copy()
    zero[0] = 0.0
    missing = y.copy()
    missing[3] = np.nan
    donor = group_codes(case.design)[0]
    constant = (np.bincount(donor, y) / np.bincount(donor))[donor]
    same = (case.samples, case.formula, case.random)
    cycle = {
        "donor": np.array(["D1", "D1", "D2", "D2", "D3", "D3"], dtype=object),
        "batch": np.array(["B1", "B2", "B2", "B3", "B3", "B1"], dtype=object),
        "x": rng.normal(0.0, 1.0, 6),
    }
    return {
        "zero_weight": Deviation(*same, y, zero),
        "missing_value": Deviation(*same, missing, ones),
        "constant_within_donors": Deviation(*same, constant, ones),
        "effects_span_every_sample": Deviation(
            cycle, "~ x", "(1 | donor) + (1 | batch)", rng.normal(0, 1, 6), np.ones(6)
        ),
    }


def cytepool_outcome(deviation: Deviation) -> tuple[str, str]:
    """Whether CytePool refuses the input, with its message, fits it, or leaves it
    unfitted."""
    design = Design(deviation.samples, deviation.formula, random=deviation.random)
    try:
        fit = fit_mixed_model(deviation.y[:, None], design, weights=deviation.weights)
    except ValueError as error:
        return "refused", str(error)
    return ("fitted" if fit.converged[0] else "unfitted"), ""


def contrast_of(case: Case) -> np.ndarray:
    """A contrast for contest1D: the last coefficient minus the one before it."""
    p = case.design.matrix.shape[1]
    contrast = np.zeros(p)
    contrast[-1] = 1.0
    contrast[-2] = -1.0
    return contrast


def write_case(case: Case, ratios: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True)
    design = case.design
    x = design.matrix
    n, p = x.shape
    case.y.T.astype("<f8").tofile(directory / "y.bin")
    x.T.astype("<f8").tofile(directory / "x.bin")
    if case.weights is not None:
        case.weights.T.astype("<f8").tofile(directory / "w.bin")
    np.sqrt(ratios).astype("<f8").tofile(directory / "theta.bin")
    contrast_of(case).astype("<f8").tofile(directory / "contrast.bin")
    groups = {
        name: [str(value) for value in design.samples[name].tolist()]
        for effect in design.random
        for name in effect.grouping
    }
    meta = {
        "n": n,
        "p": p,
        "G": case.y.shape[1],
        "random": case.random,
        "groups": groups,
        "components": [effect.group for effect in design.random],
        "weights": case.weights is not None,
        "defaults": case.defaults,
    }
    (directory / "meta.json").write_text(json.dumps(meta))


def read_doubles(path: Path) -> np.ndarray:
    """Doubles written by R, with its NA (a signalling NaN) made a quiet NaN."""
    values = np.fromfile(path, dtype="<f8")
    values[np.isnan(values)] = np.nan
    return values


# What R writes for each gene: one value, or one per column of these.
VECTORS = ("sigma2", "criterion", "given_criterion")


def read_reference(case: Case, directory: Path) -> dict[str, np.ndarray]:
    """What R wrote, by name: one row per gene."""
    n_genes = case.y.shape[1]
    reference = {}
    for path in directory.glob("lme4_*.bin"):
        name = path.stem.removeprefix("lme4_")
        values = read_doubles(path)
        reference[name] = values if name in VECTORS else values.reshape(n_genes, -1)
    return reference


def run_lme4(
    cases: list[Case], fits: list[Any], deviations: dict[str, Deviation]
) -> tuple[list[dict[str, np.ndarray]], list[str], dict[str, list[str]]]:
    rscript = os.environ.get("RSCRIPT", "Rscript")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for case, fit in zip(cases, fits):
            ratios = fit.variance_components / fit.residual_variance[:, None]
            write_case(case, ratios, root / case.name)
        (root / "cases.txt").write_text("\n".join(case.name for case in cases) + "\n")
        for name, deviation in deviations.items():
            samples = {
                key: np.asarray(values).tolist()
                for key, values in deviation.samples.items()
            }
            samples["y"] = [None if np.isnan(v) else float(v) for v in deviation.y]
            samples["w"] = [float(v) for v in deviation.weights]
            formula = f"y {deviation.formula.strip()} + {deviation.random}"
            text = json.dumps({"samples": samples, "formula": formula})
            (root / f"{name}.json").write_text(text)
        (root / "deviations.txt").write_text("\n".join(deviations) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        references = [read_reference(case, root / case.name) for case in cases]
        versions = (root / "versions.txt").read_text().split("\n")
        lines = (root / "lme4_deviations.txt").read_text().split("\n")
        outcomes = {line.split()[0]: line.split()[1:] for line in lines if line}
    return references, [v for v in versions if v], outcomes


class Table:
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

    def count(self, check: str, reference: str, metric: str, value: int) -> None:
        passed = value == 0
        self.failures += not passed
        self.rows.append(
            (check, reference, metric, str(value), "0", "pass" if passed else "FAIL")
        )

    def info(self, check: str, reference: str, metric: str, value: str) -> None:
        self.rows.append((check, reference, metric, value, "n/a", "info"))

    def deliberate(
        self, check: str, reference: str, metric: str, value: str, expected: bool
    ) -> None:
        self.failures += not expected
        result = "deliberate difference" if expected else "FAIL"
        self.rows.append((check, reference, metric, value, "n/a", result))

    def print(self) -> None:
        print("| check | reference | metric | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


def relative(
    actual: np.ndarray, desired: np.ndarray, floor: np.ndarray | float = 0.0
) -> float:
    """Largest difference relative to max(|desired|, floor); infinite if NaN differs."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    if not np.array_equal(np.isnan(actual), np.isnan(desired)):
        return np.inf
    finite = ~np.isnan(desired)
    scale = np.broadcast_to(np.fmax(np.abs(desired), floor), desired.shape)[finite]
    difference = np.abs(actual[finite] - desired[finite])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(difference == 0.0, 0.0, difference / scale)
    return float(np.max(ratios, initial=0.0))


def p_value_difference(actual: np.ndarray, desired: np.ndarray) -> float:
    """Largest difference relative to max(p, 1e-300) max(1, |log p|).

    In a tail, log p moves by about t^2 times the relative change of t, so p-values
    are compared on the scale of log p: this carries the precision of t and df into
    the tails, as the validation of voom does.
    """
    desired = np.asarray(desired)
    scale = np.fmax(desired, 1e-300) * np.fmax(
        1.0, np.abs(np.log(np.fmax(desired, 1e-300)))
    )
    return relative(actual, desired, scale)


RELATIVE = "max rel. difference"
COEFFICIENT = "max diff. / max(abs, se)"
COMPONENT = "max diff. / max(abs, sigma^2)"
STATISTIC = "max diff. / max(abs, 1)"
P_VALUE = "max diff. / (p max(1, abs log p))"


def compare(
    case: Case, fit: Any, reference: dict[str, np.ndarray], table: Table
) -> None:
    name = case.name
    tests = fixed_effect_tests(fit)
    contrast = fixed_effect_tests(fit, contrast_of(case))
    expected = reference["contrast"]
    # The optimisers locate theta to an absolute precision, so a small component has
    # little relative precision. Components are compared relative to the larger of
    # themselves and the residual variance of an observation of average weight.
    mean_weight = 1.0 if case.weights is None else case.weights.mean(axis=0)
    residual = (reference["sigma2"] / mean_weight)[:, None]
    criterion = reference["criterion"]
    ref, c = reference, contrast
    checks = [  # label, reference, metric, value, tolerance
        # The optimum, against lme4 from its own start.
        ("REML criterion at the optimum", "lme4 REMLcrit", RELATIVE,
         relative(fit.reml_criterion, criterion), EXACT),
        ("coefficients and standard errors", "lme4 fixef, vcov",
         f"{COEFFICIENT}; {RELATIVE}",
         max(relative(fit.coefficients, ref["coefficients"], ref["se"]),
             relative(tests.standard_error, ref["se"])), case.optimum),
        ("variance components and residual variance", "lme4 VarCorr, sigma",
         f"{COMPONENT}; {RELATIVE}",
         max(relative(fit.variance_components, ref["components"], residual),
             relative(fit.residual_variance, ref["sigma2"])), case.optimum),
        # Everything at CytePool's estimates: closed forms there, but lmerTest's
        # degrees of freedom come from numerical derivatives.
        ("REML criterion at CytePool's estimates", "lme4 REMLcrit", RELATIVE,
         relative(fit.reml_criterion, ref["given_criterion"]), EXACT),
        ("coefficients, standard errors and t at CytePool's estimates",
         "lme4 fixef, vcov; lmerTest", f"{COEFFICIENT}; {RELATIVE}; {STATISTIC}",
         max(relative(fit.coefficients, ref["given_coefficients"], ref["given_se"]),
             relative(tests.standard_error, ref["given_se"]),
             relative(tests.t, ref["t"], 1.0)), EXACT),
        ("Satterthwaite df", "lmerTest summary", RELATIVE,
         relative(tests.df, ref["df"]), SATTERTHWAITE),
        ("p-values", "lmerTest summary", P_VALUE,
         p_value_difference(tests.p_value, ref["p"]), SATTERTHWAITE),
        ("a contrast: estimate, standard error and t", "lmerTest contest1D",
         f"{COEFFICIENT}; {RELATIVE}; {STATISTIC}",
         max(relative(c.estimate[:, 0], expected[:, 0], expected[:, 1]),
             relative(c.standard_error[:, 0], expected[:, 1]),
             relative(c.t[:, 0], expected[:, 3], 1.0)), EXACT),
        ("a contrast: df and p-value", "lmerTest contest1D", f"{RELATIVE}; {P_VALUE}",
         max(relative(c.df[:, 0], expected[:, 2]),
             p_value_difference(c.p_value[:, 0], expected[:, 4])), SATTERTHWAITE),
    ]  # fmt: skip
    for label, source, metric, value, tolerance in checks:
        table.check(f"{name}: {label}", source, metric, value, tolerance)
    above = fit.reml_criterion - criterion > EXACT * np.abs(criterion)
    table.count(
        f"{name}: criterion above lme4's minimum",
        "lme4 REMLcrit",
        "genes",
        int(np.count_nonzero(above)),
    )
    # lme4 stops near a variance of zero rather than at it; where it does reach zero,
    # so must CytePool.
    theta = reference["theta"]
    table.count(
        f"{name}: lme4's components at zero",
        "lme4 theta = 0",
        "genes CytePool fits above zero",
        int(np.count_nonzero((~fit.boundary & (theta == 0.0)).any(axis=1))),
    )
    zero = fit.boundary
    table.info(
        f"{name}: fits with a component at zero; Newton iterations; lme4's largest "
        "theta there",
        "CytePool; lme4 getME",
        "genes; mean, max; theta",
        f"{int(np.count_nonzero(zero.any(axis=1)))} of {zero.shape[0]}; "
        f"{fit.iterations.mean():.1f}, {fit.iterations.max()}; "
        f"{np.max(theta[zero], initial=0.0):.1e}",
    )
    table.info(
        f"{name}: Satterthwaite df at lme4's optimum",
        "lmerTest summary",
        RELATIVE,
        f"{relative(tests.df, reference['df_at_optimum']):.1e}",
    )
    if case.defaults:
        components = relative(
            reference["default_components"], fit.variance_components, residual
        )
        df = relative(reference["default_df"], tests.df)
        table.info(
            f"{name}: lme4 with its default tolerances",
            "lme4 VarCorr, lmerTest summary",
            f"components: {COMPONENT}; df: {RELATIVE}",
            f"{components:.1e}; {df:.1e}",
        )


def high_precision_fit(case: Case, fit: Any, gene: int) -> tuple[np.ndarray, ...]:
    """The free thetas of one gene and the df of each coefficient, from definitions.

    With 30 digits (mpmath): the REML deviance in (gamma_k, sigma^2) from dense
    matrices, with the components at zero held there, minimised by Newton's method on
    the profiled criterion from CytePool's estimates, and differentiated numerically.
    """
    import mpmath

    mpmath.mp.dps = 30
    design = case.design
    n, p = (int(size) for size in design.matrix.shape)
    x = mpmath.matrix(design.matrix.tolist())
    y = mpmath.matrix(case.y[:, gene].tolist())
    weights = np.ones(n) if case.weights is None else case.weights[:, gene]
    noise = mpmath.diag([1 / mpmath.mpf(float(w)) for w in weights])
    ratios = fit.variance_components[gene] / fit.residual_variance[gene]
    free = np.flatnonzero(ratios > 0.0)
    shapes = []
    for k in free:
        codes = group_codes(design)[k]
        z = mpmath.matrix(n, int(codes.max()) + 1)
        for i, code in enumerate(codes.tolist()):
            z[i, code] = 1
        shapes.append(z * z.T)
    size = len(free)

    def parts(gamma: list[Any]) -> tuple[Any, ...]:
        v = noise + sum((g * s for g, s in zip(gamma, shapes)), mpmath.zeros(n, n))
        inverse = mpmath.inverse(v)
        information = x.T * inverse * x
        bread = mpmath.inverse(information)
        py = inverse * y - inverse * x * (bread * (x.T * inverse * y))
        return v, information, bread, (y.T * py)[0]

    def deviance(*arguments: Any) -> Any:
        *gamma, s2 = arguments
        v, information, _, r2 = parts(gamma)
        log_det = mpmath.log(mpmath.det(v)) + mpmath.log(mpmath.det(information))
        return (n - p) * mpmath.log(s2) + log_det + r2 / s2

    def profiled(*gamma: Any) -> Any:
        return deviance(*gamma, parts(list(gamma))[3] / (n - p))

    def order(a: int, b: int, count: int) -> tuple[int, ...]:
        return tuple((a == i) + (b == i) for i in range(count))

    gamma = [mpmath.mpf(float(ratios[k])) for k in free]
    for _ in range(3):
        if not size:
            break
        gradient = mpmath.matrix(
            [mpmath.diff(profiled, gamma, order(a, -1, size)) for a in range(size)]
        )
        hessian = mpmath.matrix(size, size)
        for a in range(size):
            for b in range(size):
                hessian[a, b] = mpmath.diff(profiled, gamma, order(a, b, size))
        gamma = list(mpmath.matrix(gamma) - mpmath.lu_solve(hessian, gradient))
    point = [*gamma, parts(gamma)[3] / (n - p)]
    hessian = mpmath.matrix(size + 1, size + 1)
    for a in range(size + 1):
        for b in range(size + 1):
            hessian[a, b] = mpmath.diff(deviance, point, order(a, b, size + 1))
    covariance = 2 * mpmath.inverse(hessian)
    df = np.empty(p)
    for j in range(p):

        def variance(*arguments: Any, j: int = j) -> Any:
            *gamma, s2 = arguments
            return s2 * parts(gamma)[2][j, j]

        gradient = mpmath.matrix(
            [
                mpmath.diff(variance, point, order(a, -1, size + 1))
                for a in range(size + 1)
            ]
        )
        spread = (gradient.T * covariance * gradient)[0]
        df[j] = float(2 * variance(*point) ** 2 / spread)
    return np.array([float(mpmath.sqrt(g)) for g in gamma]), df


def write_fixtures(
    cases: list[Case], fits: list[Any], references: list[dict[str, np.ndarray]]
) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for case, reference in zip(cases, references):
        arrays: dict[str, np.ndarray] = {
            "y": case.y,
            "formula": np.array(case.formula),
            "random": np.array(case.random),
            "contrast_weights": contrast_of(case),
            # Text columns as unicode arrays, which npz files store without pickle.
            **{
                f"sample_{key}": np.asarray(
                    value, dtype=str if np.asarray(value).dtype == object else None
                )
                for key, value in case.samples.items()
            },
            **{
                key: value
                for key, value in reference.items()
                if not key.startswith("default") and key != "df_at_optimum"
            },
        }
        if case.weights is not None:
            arrays["weights"] = case.weights
        np.savez_compressed(FIXTURE_DIRECTORY / f"{case.name}.npz", **arrays)
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIRECTORY}")


def benchmark(n_genes: int = 20_000, n_timed: int = 100) -> None:
    """Time fit_mixed_model on one thread, and lmerTest on ``n_timed`` of the genes."""
    rng = np.random.default_rng(20261028)
    donor = np.repeat(np.arange(40), 3)
    samples = {
        "donor": labels("D", donor),
        "batch": labels("B", rng.integers(0, 6, donor.size)),
        "condition": np.tile(np.array(["a", "b", "c"], dtype=object), 40),
        "age": rng.normal(50.0, 10.0, donor.size),
    }
    # 20 donors x 3 samples x 2 replicates, with sample labels repeated across donors.
    nested = samples | {
        "donor": labels("D", donor // 2),
        "sample": labels("S", np.arange(donor.size) // 2 % 3),
    }
    formula = "~ condition + age"
    cases = [
        Case("one_intercept", samples, formula, "(1 | donor)", None),
        Case("crossed", samples, formula, "(1 | donor) + (1 | batch)", None),
        Case("nested", nested, formula, "(1 | donor/sample)", None),
    ]
    for case in cases:
        y = simulate(rng, case.design, n_genes, random_sd=2.0)
        case.y, case.weights = with_weights(rng, y)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for case in cases:
            assert case.weights is not None
            timed = Case(case.name, case.samples, case.formula, case.random,
                         case.y[:, :n_timed], case.weights[:, :n_timed])  # fmt: skip
            components = len(case.design.random)
            write_case(timed, np.zeros((n_timed, components)), root / case.name)
        (root / "cases.txt").write_text("\n".join(case.name for case in cases) + "\n")
        (root / "benchmark.R").write_text(R_BENCHMARK)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "benchmark.R"), str(root)], check=True
        )
        lines = (root / "seconds.txt").read_text().splitlines()
        theirs = {line.split()[0]: float(line.split()[1]) for line in lines}
    n = cases[0].y.shape[0]
    print(f"Benchmark: {n_genes} genes x {n} samples, observation weights, one thread")
    print(
        "| random effects | CytePool fit (s) | CytePool tests (s) | lmerTest per gene "
        "(s) | lmerTest for every gene (min) |"
    )
    print("| --- | --- | --- | --- | --- |")
    for case in cases:
        warm = fit_mixed_model(
            case.y[:, :10], case.design, weights=case.weights[:, :10]
        )
        fixed_effect_tests(warm)
        seconds = []
        for _ in range(3):
            start = time.perf_counter()
            fit = fit_mixed_model(
                case.y, case.design, weights=case.weights, n_threads=1
            )
            seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        fixed_effect_tests(fit)
        tests = time.perf_counter() - start
        per_gene = theirs[case.name]
        random = case.random.replace("|", "\\|")
        print(
            f"| {random} | {np.median(seconds):.2f} | {tests:.2f} | "
            f"{per_gene:.3f} | {per_gene * n_genes / 60:.1f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="time 20,000 genes x 120 samples"
    )
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark()
        return 0
    cases = make_cases(N_GENES, N_MANY, seed=20261025)
    fixture_cases = []
    if arguments.write_fixtures:
        fixture_cases = [
            case
            for case in make_cases(N_FIXTURE_GENES, N_FIXTURE_GENES, seed=20261026)
            if case.name != "many_genes"
        ]
        for case in fixture_cases:
            case.name = f"fixture_{case.name}"
            case.defaults = False
    everything = cases + fixture_cases
    fits = [
        fit_mixed_model(case.y, case.design, weights=case.weights)
        for case in everything
    ]
    deviations = deviation_cases(cases[0], np.random.default_rng(20261027))
    references, versions, outcomes = run_lme4(everything, fits, deviations)
    print("Reference: " + ", ".join(versions))
    table = Table()
    for case, fit, reference in zip(everything, fits, references):
        compare(case, fit, reference, table)
    # For the cases with one random intercept, and for those with several, the gene
    # whose df differ most from lmerTest's, from the definition with 30 digits: which
    # of the two is exact.
    for several in (False, True):
        worst = (-1.0, 0, 0)
        for index, (case, fit, reference) in enumerate(
            zip(everything, fits, references)
        ):
            if (len(case.design.random) > 1) == several:
                df = fixed_effect_tests(fit).df
                for g in range(df.shape[0]):
                    worst = max(worst, (relative(df[g], reference["df"][g]), index, g))
        _, index, gene = worst
        case, fit, reference = everything[index], fits[index], references[index]
        _, exact = high_precision_fit(case, fit, gene)
        table.check(
            f"{case.name}, gene {gene}: Satterthwaite df of the gene that differs most "
            "from lmerTest",
            "the definition, 30 digits (mpmath)",
            RELATIVE,
            relative(fixed_effect_tests(fit).df[gene], exact),
            EXACT,
        )
        table.info(
            f"{case.name}, gene {gene}: lmerTest's df there",
            "the definition, 30 digits (mpmath)",
            RELATIVE,
            f"{relative(reference['df'][gene], exact):.1e}",
        )
    # The gene of the large-ratio case whose variance components differ most from
    # lme4's: its thetas from the definition with 30 digits.
    index = next(i for i, case in enumerate(everything) if case.name == "large_ratios")
    case, fit, reference = everything[index], fits[index], references[index]
    differences = np.abs(fit.variance_components - reference["components"])
    gene = int(np.argmax(np.max(differences / fit.residual_variance[:, None], axis=1)))
    free = ~fit.boundary[gene]
    theta = np.sqrt(fit.variance_components[gene] / fit.residual_variance[gene])
    exact, _ = high_precision_fit(case, fit, gene)
    table.check(
        f"large_ratios, gene {gene}: theta of the gene that differs most from lme4",
        "the optimum with 30 digits (mpmath)",
        RELATIVE,
        relative(theta[free], exact),
        EXACT,
    )
    table.info(
        f"large_ratios, gene {gene}: lme4's theta there",
        "the optimum with 30 digits (mpmath)",
        RELATIVE,
        f"{relative(reference['theta'][gene][free], exact):.1e}",
    )
    # Inputs where CytePool deliberately differs from lme4, and what each program does.
    lme4 = {
        name: (float(criterion), int(used), float(sigma), warned == "TRUE")
        for name, (criterion, used, sigma, warned) in outcomes.items()
    }
    n = cases[0].y.shape[0]
    criterion, used = lme4["zero_weight"][0], lme4["missing_value"][1]
    _, _, sigma, warned = lme4["constant_within_donors"]
    sigma = sigma / float(np.std(deviations["constant_within_donors"].y))
    spanning = lme4["effects_span_every_sample"][0]
    rows = [  # name, label, lme4's metric and value, whether it is as described,
        # CytePool's outcome and the reason it gives
        ("zero_weight", "a zero weight", "REMLcrit", f"{criterion:.4g}",
         criterion == np.inf, "refused", "weights must be finite and positive"),
        ("missing_value", "a missing value", "samples used", f"{used} of {n}",
         used == n - 1, "refused", "y must be finite"),
        ("constant_within_donors", "values constant within every donor",
         "sigma / sd(y), warned", f"{sigma:.1e}, {warned}", sigma < 1e-3 and warned,
         "unfitted", ""),
        ("effects_span_every_sample", "fixed and random effects that span every "
         "sample", "REMLcrit", f"{spanning:.4g}", np.isfinite(spanning), "refused",
         "together span every sample"),
    ]  # fmt: skip
    for name, label, metric, value, described, ours, reason in rows:
        outcome, message = cytepool_outcome(deviations[name])
        table.deliberate(
            f"deviation: {label}",
            "lme4 lmer",
            f"lme4 {metric}; CytePool",
            f"{value}; {outcome}",
            bool(described) and outcome == ours and reason in message,
        )
    table.print()
    if arguments.write_fixtures:
        if table.failures:
            print("not writing fixtures: some checks failed")
        else:
            for case in fixture_cases:
                case.name = case.name.removeprefix("fixture_")
            write_fixtures(fixture_cases, fits[len(cases) :], references[len(cases) :])
    print(f"{len(table.rows)} checks, {table.failures} failed")
    return 1 if table.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
