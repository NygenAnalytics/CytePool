"""Check combined p-values, partial conjunctions, pi0 and replication across studies.

References, each run as a separate program on files in a temporary directory:

- scipy's ``combine_pvalues`` (``fisher``, ``pearson``, ``tippett``, ``stouffer`` with
  weights, ``mudholkar_george``), feature by feature on the p-values each feature has;
  mpmath at 400 digits for 13 features of p-values from 5e-324 to 1 - 1e-15. Both run
  in this script started again as ``--reference DIRECTORY``, which does not import
  CytePool.
- metap 1.1: ``sumlog``, ``sumz`` (with weights), ``minimump``, ``wilkinsonp``
  (r = 2), ``maximump``, ``logitp``, and Simes's combination as the smallest
  Benjamini-Hochberg p-value of ``p.adjust``; with directions, each direction's
  one-sided p-values from ``two2one``, doubled at the smaller. ``two2one`` rounds
  1 - p / 2 to a double, which loses the digits of p / 2 for a small p, so these rows
  hold the features whose p-values are all at least 1e-5, and mpmath checks small
  ones. The partial conjunction p-values from their definition (Benjamini and Heller
  2008): ``sumlog``, ``sumz`` or Simes's combination of the n - u + 1 largest
  p-values, sorted in R.
- qvalue 2.38.0: ``pi0est`` with its defaults, ``smooth.df = 5``, ``smooth.log.pi0``,
  ``pi0.method = "bootstrap"``, one lambda and three other grids, on five sets of
  p-values (one rounded to 0.01, so that p-values equal values of lambda), and its
  behaviour where CytePool differs.
- metafor 5.0.1: ``rma.uni`` for the heterogeneity of every gene, REML (with
  ``control = list(threshold = 1e-13, maxiter = 10000)``) or DerSimonian-Laird.

Cases (seed ``SEED``): p-values of 2, 5 and 20 studies for 300, 400 and 200 features,
a third with effects, none, 15% and 5% missing, with signs and with weights (square
roots of sample sizes). Then three cohorts of 6, 4 and 8 donors, before and after a
stimulation that changes 15% of 700 genes in two cell types alike in every cohort, up
to 50% more or less, each measuring part of the genes, are aligned in a
``CubeCollection``, tested by ``limma_voom`` and passed to ``replication`` with four
settings, among them pi1 with qvalue's defaults (``null_proportion={}``) and pi1 by
default (``null_proportion`` left out), which is ``pi0est(lambda = 0.5)``; R computes
every column from the cohorts' tables, and the figures are compared with what R
computes for them.

Tolerances: exact (1e-10) for every closed form, the smoother of pi0 included (R's
search for its smoothing parameter is followed step by step); close (1e-6) for REML.
Differences are scaled by max(|reference|, 1), and p-values by
max(p, 1e-300) max(1, |log p|); counts and directions must be equal.

Requirements:
    Python with numpy, numba, scipy, mpmath and matplotlib, run from the repository
    root so that ``cytepool`` imports from the working tree. R 4.4 with metap, qvalue
    and jsonlite (``RSCRIPT``), and with metafor and jsonlite (``METAFOR_RSCRIPT``, by
    default ``RSCRIPT``). R runs with its address space limited to 8 GB.

Usage:
    PYTHONPATH=. python validation/combined_p_values_and_replication.py
    PYTHONPATH=. python validation/combined_p_values_and_replication.py --write-fixtures
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SEED = 20261102
EXACT, CLOSE = 1e-10, 1e-6
MEMORY = 8_000_000 * 1024  # bytes of address space for R
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "data"
FIXTURES /= "combined_p_values_and_replication"
SCIPY = ("fisher", "pearson", "tippett", "stouffer", "mudholkar_george")
# metap's methods and CytePool's (Wilkinson's r = 2, the maximum r = k).
METAP = {"fisher": "fisher", "stouffer": "stouffer", "tippett": "tippett"}
METAP |= {"wilkinson": "wilkinson", "maximum": "wilkinson"}
METAP |= {"logit": "mudholkar_george", "simes": "simes"}
EXTREME = ("fisher", "pearson", "stouffer", "mudholkar_george", "tippett")
EXTREME += ("wilkinson", "simes")
CONJUNCTIONS = ("simes", "fisher", "stouffer")
SMALLEST_NORMAL = 2.2250738585072014e-308
# pi0est's options, and CytePool's names of them.
PI0_OPTIONS = {
    "defaults": {},
    "smooth.df = 5": {"smooth.df": 5},
    "smooth.log.pi0": {"smooth.log.pi0": True},
    "bootstrap": {"pi0.method": "bootstrap"},
    "lambda = 0.5": {"lambda": 0.5},
    "lambda = 0.1, ..., 0.9": {"lambda": [0.1 * j for j in range(1, 10)]},
    "lambda unsorted": {"lambda": [0.5, 0.2, 0.8, 0.35, 0.65]},
    "lambda = 0, ..., 0.9, bootstrap": {
        "lambda": [0.05 * j for j in range(19)],
        "pi0.method": "bootstrap",
    },
}
NAMES = {"smooth.df": "smooth_df", "smooth.log.pi0": "smooth_log"}
NAMES |= {"pi0.method": "method", "lambda": "lambdas"}
# The replications: method, conjunction method, at_least, alpha, meta_method, and
# pi0est's options; None leaves null_proportion out, for pi1 by default, which is
# pi0est at DEFAULT_PI0.
SETTINGS = (
    ("stouffer", "simes", 2, 0.05, "reml", {}),
    ("fisher", "fisher", 3, 0.25, "dl", {"pi0.method": "bootstrap"}),
    ("tippett", "stouffer", 2, 0.1, "reml", {"lambda": 0.5}),
    ("stouffer", "simes", 2, 0.05, "dl", None),
)
DEFAULT_PI0 = {"lambda": 0.5}


def reference_main(directory):
    """scipy's and mpmath's values of the inputs in ``directory``."""
    import mpmath as mp
    import scipy
    from scipy.stats import combine_pvalues

    mp.mp.dps = 400  # 1 - p / 2 of the smallest p-values
    data = dict(np.load(directory / "inputs.npz"))
    out = {}
    for name in {key.split("/")[0] for key in data if not key.startswith("extreme")}:
        p, w = data[f"{name}/p"], data[f"{name}/weights"]
        for method in SCIPY:
            values = np.full((2, p.shape[1]), np.nan)
            for j in range(p.shape[1]):
                keep = ~np.isnan(p[:, j])
                if keep.any():
                    extra = {"weights": w[keep]} if method == "stouffer" else {}
                    r = combine_pvalues(p[keep, j], method=method, **extra)
                    values[:, j] = r.statistic, r.pvalue
            out[f"{name}/{method}"] = values

    def quantile(p):  # z with P(Z > z) = p
        if p in (0, 1):
            return mp.inf if p == 0 else -mp.inf
        if p > 0.5:
            return -quantile(1 - p)
        log_p = mp.log(p)
        upper = lambda z: mp.log(mp.erfc(z / mp.sqrt(2)) / 2) - log_p  # noqa: E731
        return mp.findroot(upper, mp.sqrt(-2 * log_p))

    def combined(q, method, w):
        k, ordered = len(q), sorted(q)
        if method in ("fisher", "pearson"):
            logs = [mp.log(v) if method == "fisher" else mp.log1p(-v) for v in q]
            x = -2 * mp.fsum(logs)
            tail = (x / 2, mp.inf) if method == "fisher" else (0, x / 2)
            return x, mp.gammainc(k, *tail, regularized=True)
        if method == "stouffer":
            z = mp.fsum(a * quantile(v) for a, v in zip(w, q, strict=True))
            z /= mp.sqrt(mp.fsum(a**2 for a in w))
            return z, mp.erfc(z / mp.sqrt(2)) / 2
        if method == "mudholkar_george":
            t = mp.fsum(mp.log1p(-v) - mp.log(v) for v in q) / mp.pi
            t *= mp.sqrt(mp.mpf(3) / k) * mp.sqrt(mp.mpf(5 * k + 4) / (5 * k + 2))
            nu = mp.mpf(5 * k + 4)
            tail = mp.betainc(nu / 2, 0.5, 0, nu / (nu + t**2), regularized=True)
            return t, tail / 2 if t >= 0 else 1 - tail / 2
        if method == "tippett":
            return ordered[0], -mp.expm1(k * mp.log1p(-ordered[0]))
        if method == "wilkinson":
            return ordered[1], mp.betainc(2, k - 1, 0, ordered[1], regularized=True)
        s = min(mp.mpf(k) * v / (j + 1) for j, v in enumerate(ordered))  # Simes
        return s, s

    p, signs, w = (data[f"extreme/{part}"] for part in ("p", "signs", "weights"))
    for method in EXTREME:
        values = np.full((4, p.shape[1]), np.nan)  # p, statistic; signed p, statistic
        for j in range(p.shape[1]):
            keep = np.flatnonzero(~np.isnan(p[:, j]))
            if method == "wilkinson" and keep.size < 2:
                continue
            q = [mp.mpf(float(v)) for v in p[keep, j]]
            weights = [mp.mpf(float(v)) for v in w[keep]]
            values[1::-1, j] = [float(v) for v in combined(q, method, weights)]
            up = [v / 2 if s > 0 else 1 - v / 2 for v, s in zip(q, signs[keep, j])]
            down = [1 - v for v in up]
            pair = [combined(side, method, weights)[1] for side in (up, down)]
            values[2, j] = float(min(1, 2 * min(pair)))
            values[3, j] = float(combined(up, method, weights)[0])
        out[f"extreme/{method}"] = values
    np.savez(directory / "reference.npz", **out)
    versions = {"scipy": scipy.__version__, "mpmath": mp.__version__}
    (directory / "versions.json").write_text(json.dumps(versions))


R_META = r"""
suppressPackageStartupMessages({library(metap); library(qvalue)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, path) writeBin(as.double(x), path, size = 8)
shape <- function(dir) as.integer(readLines(file.path(dir, "shape.txt")))
matrix_of <- function(dir, file) matrix(rd(file.path(dir, file)), shape(dir)[1])
simes <- function(p) min(p.adjust(p, method = "BH"))
# metap's statistic and p-value; NA where it stops, for fewer than two p-values.
combine <- function(p, w, method) {
  if (length(p) < 2) return(c(NA, NA))
  switch(method,
    fisher = { r <- sumlog(p); c(r$chisq, r$p) },
    stouffer = { r <- sumz(p, weights = w); c(r$z, r$p) },
    tippett = { r <- minimump(p); c(r$pr, r$p) },
    wilkinson = { r <- wilkinsonp(p, r = 2); c(r$pr, r$p) },
    maximum = { r <- maximump(p); c(r$pr, r$p) },
    logit = { r <- logitp(p); c(r$t, r$p) },
    simes = { v <- simes(p); c(v, v) })
}
# The partial conjunction p-value from its definition: the n - u + 1 largest.
conjunction <- function(p, u, method) {
  if (length(p) < u) return(NA)
  largest <- sort(p)[u:length(p)]
  if (length(largest) == 1) return(largest)
  switch(method, simes = simes(largest), fisher = sumlog(largest)$p,
         stouffer = sumz(largest)$p)
}
# two2one's one-sided p-values; metap leaves out those that round to 1, for a
# two-sided p-value below 2.2e-16 in the other direction, so they are set to the
# largest double below 1, which moves only the larger of the two directions.
oneside <- function(p, s, up) pmin(two2one(p, invert = if (up) s < 0 else s > 0), 1 - 1e-16)
signed <- function(f, p, s) min(1, 2 * min(f(oneside(p, s, TRUE)), f(oneside(p, s, FALSE))))
for (case in readLines(file.path(root, "combine", "cases.txt"))) {
  dir <- file.path(root, "combine", case)
  p <- matrix_of(dir, "p.bin"); s <- matrix_of(dir, "signs.bin"); w <- rd(file.path(dir, "weights.bin"))
  methods <- strsplit(readLines(file.path(dir, "methods.txt")), " ")[[1]]
  out <- sapply(seq_len(ncol(p)), function(j) {
    k <- which(!is.na(p[, j])); q <- p[k, j]; d <- s[k, j]
    v <- NULL
    for (m in methods)
      v <- c(v, combine(q, w[k], m), if (length(q) < 2) NA else
             signed(function(x) combine(x, w[k], m)[2], q, d))
    for (u in 1:3) for (m in c("simes", "fisher", "stouffer"))
      v <- c(v, conjunction(q, u, m), if (length(q) < u) NA else
             signed(function(x) conjunction(x, u, m), q, d))
    v
  })
  wr(out, file.path(dir, "metap.bin"))
}
pi0_results <- list()
for (run in readLines(file.path(root, "pi0", "runs.txt"))) {
  part <- strsplit(run, "\t")[[1]]  # the p-values' file, the options as JSON
  args <- c(list(p = rd(file.path(root, "pi0", part[1]))), jsonlite::fromJSON(part[2]))
  r <- tryCatch(do.call(pi0est, args), error = function(e) conditionMessage(e))
  pi0_results[[run]] <- if (is.character(r)) list(error = r) else
    list(pi0 = r$pi0, pi0_lambda = r$pi0.lambda, lambda = r$lambda,
         pi0_smooth = if (is.null(r$pi0.smooth)) NA else r$pi0.smooth)
}
writeLines(jsonlite::toJSON(pi0_results, digits = I(17), auto_unbox = TRUE, na = "null"),
           file.path(root, "pi0", "qvalue.json"))
for (analysis in readLines(file.path(root, "replication", "analyses.txt"))) {
  spec <- jsonlite::fromJSON(analysis)
  dir <- file.path(root, "replication", spec$name)
  lfc <- matrix_of(dir, "lfc.bin"); p <- matrix_of(dir, "p.bin"); adj <- matrix_of(dir, "adj.bin")
  w <- if (spec$method == "stouffer") sqrt(rd(file.path(dir, "n_samples.bin"))) else NULL
  tested <- !is.na(p); found <- tested & adj < spec$alpha; found[is.na(found)] <- FALSE
  genes <- sapply(seq_len(ncol(p)), function(j) {
    k <- which(tested[, j]); q <- p[k, j]; s <- sign(lfc[k, j])
    # One study's one-sided p-value is its own combination, which metap refuses.
    f <- function(x) if (length(x) == 1) x else combine(x, w[k], spec$method)[2]
    up <- f(oneside(q, s, TRUE)); down <- f(oneside(q, s, FALSE))
    direction <- sign(down - up)
    replicated <- if (length(q) < spec$at_least) NA else
      signed(function(x) conjunction(x, spec$at_least, spec$conjunction), q, s)
    c(length(q), sum(s > 0), sum(s < 0), sum(found[k, j]), sum(found[k, j] & s > 0),
      sum(found[k, j] & s < 0), direction, if (direction == 0) NA else mean(s == direction),
      min(1, 2 * min(up, down)), replicated)
  })
  wr(rbind(genes, p.adjust(genes[9, ], "BH"), p.adjust(genes[10, ], "BH")), file.path(dir, "genes.bin"))
  pairs <- NULL
  for (a in seq_len(nrow(p))) for (b in seq_len(nrow(p))) if (a != b) {
    chosen <- found[a, ] & tested[b, ]; pi1 <- NA; agreement <- NA
    if (any(chosen)) {
      r <- tryCatch(do.call(pi0est, c(list(p = p[b, chosen]), spec$pi0)), error = function(e) NULL)
      if (!is.null(r)) pi1 <- 1 - r$pi0
      agreement <- mean(sign(lfc[a, chosen]) == sign(lfc[b, chosen]))
    }
    pairs <- cbind(pairs, c(a, b, sum(chosen), sum(chosen & found[b, ]), agreement, pi1))
  }
  wr(pairs, file.path(dir, "pairs.bin"))
  same <- pmax(genes[5, ], genes[6, ])
  wr(sapply(seq_len(nrow(p)), function(j) c(j, sum(genes[4, ] >= j), sum(same >= j),
     sum(genes[4, ] == j & same == j), sum(genes[4, ] == j & same < j))), file.path(dir, "found_in.bin"))
}
# pi0est and metap where CytePool differs, or must refuse alike.
probe <- function(expr) tryCatch(list(value = expr), error = function(e) list(error = conditionMessage(e)))
u <- rd(file.path(root, "probe.bin"))
zero_warning <- ""
probes <- list(
  zero_count_smoother = probe(pi0est(u * 0.9)$pi0),
  zero_count_bootstrap = probe(pi0est(u * 0.9, pi0.method = "bootstrap")$pi0),
  zero_count_log = probe(pi0est(u * 0.9, smooth.log.pi0 = TRUE)$pi0),
  estimate_zero = probe(pi0est(u * 0.4, lambda = 0.5)$pi0),
  missing_p = probe(pi0est(c(u^2, NA))$pi0),
  two_lambdas = probe(pi0est(u, lambda = c(0.2, 0.5))$pi0),
  lambda_one = probe(pi0est(u, lambda = 1)$pi0),
  lambdas_96 = probe(pi0est(u^1.3, lambda = seq(0, 0.95, 0.01))$pi0),
  metap_zero = probe(withCallingHandlers(sumlog(c(0, 0.2, 0.5))$p, warning = function(w) {
    zero_warning <<- conditionMessage(w); invokeRestart("muffleWarning")})),
  metap_one_study = probe(sumlog(0.2)$p))
probes$metap_zero$warning <- zero_warning
writeLines(jsonlite::toJSON(probes, digits = I(17), auto_unbox = TRUE), file.path(root, "probes.json"))
writeLines(c(paste("R", getRversion()), paste("metap", packageVersion("metap")),
             paste("qvalue", packageVersion("qvalue")), paste("jsonlite", packageVersion("jsonlite"))),
           file.path(root, "versions_meta.txt"))
"""

R_METAFOR = r"""
suppressPackageStartupMessages(library(metafor))
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
for (analysis in readLines(file.path(root, "replication", "analyses.txt"))) {
  spec <- jsonlite::fromJSON(analysis)
  dir <- file.path(root, "replication", spec$name)
  k <- as.integer(readLines(file.path(dir, "shape.txt")))[1]
  y <- matrix(rd(file.path(dir, "lfc.bin")), k); se <- matrix(rd(file.path(dir, "se.bin")), k)
  out <- sapply(seq_len(ncol(y)), function(j) {
    keep <- !is.na(y[, j])
    if (sum(keep) < 2) return(rep(NA_real_, 4))
    r <- rma.uni(y[keep, j], sei = se[keep, j], method = toupper(spec$meta),
                 control = list(threshold = 1e-13, maxiter = 10000))
    c(r$QE, r$QEp, r$I2 / 100, r$tau2)
  })
  writeBin(as.double(out), file.path(dir, "heterogeneity.bin"), size = 8)
}
writeLines(paste("metafor", packageVersion("metafor")), file.path(root, "versions_metafor.txt"))
"""


def p_value_cases(rng):
    """p-values (studies x features), NaN where missing, signs and weights."""

    def draw(k, n, missing):
        effect = rng.random(n) < 1 / 3
        p = np.where(effect, rng.beta(0.25, 4.0, (k, n)), rng.uniform(size=(k, n)))
        signs = np.where(rng.random((k, n)) < np.where(effect, 0.85, 0.5), 1.0, -1.0)
        gone = rng.random((k, n)) < missing
        p[gone], signs[gone] = np.nan, np.nan
        return p, signs, np.sqrt(rng.integers(4, 60, k).astype(float))

    cases = {"2 studies": draw(2, 300, 0.0), "5 studies": draw(5, 400, 0.15)}
    p, signs, _ = cases["5 studies"]
    p[1:, :4], signs[1:, :4] = np.nan, np.nan  # four features of one study
    p[:, 4:6], signs[:, 4:6] = np.nan, np.nan  # two of none
    cases["20 studies"] = draw(20, 200, 0.05)
    return cases


def extreme_case():
    """Features of extreme p-values, with signs and weights, for mpmath."""
    columns = [
        [1e-300, 0.5, 0.3],
        [1e-300, 1e-300, 1e-300],
        [1e-200, 1e-250, 1e-10, 0.9],
        [5e-324, 0.5],
        [2e-308, 0.7, 1e-305],
        [1e-17, 1e-16, 3e-17],
        [1 - 1e-12, 1 - 1e-15, 0.999999],
        [0.5, 0.5, 0.5],
        [1e-300],
        [1e-100] * 10,
        [1e-50, 0.999999999, 1e-60, 0.2],
        [0.04, 0.03, 0.02, 1e-30, 0.9, 0.5],
        [1e-12, 1e-12],
    ]
    p = np.full((max(map(len, columns)), len(columns)), np.nan)
    signs = np.full_like(p, np.nan)
    for j, column in enumerate(columns):
        p[: len(column), j] = column
        signs[: len(column), j] = [(-1.0) ** (i * j) for i in range(len(column))]
    return p, signs, np.sqrt(np.arange(5.0, 5.0 + p.shape[0]))


def probe_p_values():
    """The p-values of the probes of pi0est's behaviour."""
    return np.random.default_rng(1).uniform(size=500)


def pi0_sets(rng):
    """p-values of different numbers of tests and shares of true nulls."""
    sets = {}
    for m, pi0, shape in (
        (1000, 0.8, (0.2, 3.0)),
        (20000, 0.9, (0.3, 5.0)),
        (200, 0.5, (0.4, 2.0)),
        (60, 0.3, (0.1, 4.0)),
        (5000, 0.95, (0.5, 3.0)),
    ):
        n0 = round(m * pi0)
        p = np.concatenate([rng.uniform(size=n0), rng.beta(*shape, size=m - n0)])
        sets[f"m = {m:,}, pi0 = {pi0:g}"] = p
    sets["m = 5,000, pi0 = 0.95"] = np.round(sets["m = 5,000, pi0 = 0.95"], 2)
    return sets


def cohorts(rng, n_genes, n_donors=(6, 4, 8)):
    """Three cohorts' cubes, their collection and limma_voom results."""
    from cytepool import Cube, CubeCollection, Design, limma_voom

    genes = [f"g{j:04d}" for j in range(n_genes)]
    base = {g: rng.normal(0.0, 1.5, n_genes) for g in "TB"}
    change = {
        g: rng.normal(0.0, 1.0, n_genes) * (rng.random(n_genes) < 0.15) for g in "TB"
    }
    tenth = n_genes // 10
    measured = {
        "A": genes[: 9 * tenth],
        "B": genes[tenth:],
        "C": genes[::2] + genes[1::4],
    }
    cubes = {}
    for (name, ids), donors in zip(measured.items(), n_donors, strict=True):
        where = np.array([genes.index(g) for g in ids])
        samples = {
            "sample_id": [
                f"{name}{d}_{c}" for d in range(donors) for c in ("ctrl", "stim")
            ],
            "donor": [f"{name}{d}" for d in range(donors) for _ in range(2)],
            "condition": ["ctrl", "stim"] * donors,
        }
        counts = np.zeros((2, 2 * donors, where.size), dtype=np.uint32)
        for g, kind in enumerate("TB"):
            # The cohort's own departure from the common change: heterogeneity.
            own = change[kind][where] * rng.uniform(0.5, 1.5, where.size)
            donor = {d: rng.normal(0.0, 0.25, where.size) for d in samples["donor"]}
            for s in range(2 * donors):
                log_mean = base[kind][where] + donor[samples["donor"][s]]
                log_mean = log_mean + own * (samples["condition"][s] == "stim")
                mean = rng.uniform(1e5, 4e5) * np.exp(log_mean) / np.exp(log_mean).sum()
                counts[g, s] = rng.negative_binomial(20.0, 20.0 / (20.0 + mean))
        cubes[name] = Cube(
            {"counts": counts},
            samples=samples,
            groups={"group_id": ["T", "B"]},
            features={"feature_id": ids},
            n_cells=rng.integers(40, 90, (2 * donors, 2)),
        )
    collection = CubeCollection(cubes, align_groups_by="group_id")
    results = {}
    for name in collection.studies:
        cube = collection.aligned_cube(name)
        levels = {"condition": ["ctrl", "stim"]}
        contrasts = {"stim_vs_ctrl": "conditionstim"}
        design = Design(
            cube.samples, "~ donor + condition", levels=levels, contrasts=contrasts
        )
        results[name] = limma_voom(cube, design)
    return cubes, collection, results


def replications(collection, results, settings=SETTINGS):
    """replication of every group with every setting."""
    from cytepool import StudyResults, replication

    studies = StudyResults(collection, results)
    out = []
    for group in collection.groups["group_id"].tolist():
        for s, (method, conjunction, u, alpha, meta, pi0) in enumerate(settings):
            options = {} if pi0 is None else {
                "null_proportion": {NAMES[key]: value for key, value in pi0.items()}
            }  # fmt: skip
            result = replication(
                studies, group, "stim_vs_ctrl", alpha=alpha, method=method,
                at_least=u, conjunction_method=conjunction, meta_method=meta,
                **options,
            )  # fmt: skip
            out.append((f"{group}{s}", s, result))
    return out


def write_binary(path, values):
    path.write_bytes(np.ascontiguousarray(values, dtype="<f8").tobytes())


def run_references(directory, cases, extreme, pi0, analyses, collection, results):
    """Write the inputs, run the reference programs and read their outputs."""
    from cytepool import StudyResults

    parts = ("p", "signs", "weights")
    inputs = {
        f"extreme/{part}": value for part, value in zip(parts, extreme, strict=True)
    }
    for name, values in cases.items():
        inputs |= {
            f"{name}/{part}": value for part, value in zip(parts, values, strict=True)
        }
    np.savez(directory / "inputs.npz", **inputs)
    subprocess.run(
        [sys.executable, __file__, "--reference", str(directory)], check=True
    )
    for c, (p, signs, w) in enumerate(cases.values()):
        folder = directory / "combine" / f"case{c}"
        folder.mkdir(parents=True)
        (folder / "shape.txt").write_text(f"{p.shape[0]}\n{p.shape[1]}\n")
        write_binary(folder / "p.bin", p.T)
        write_binary(folder / "signs.bin", signs.T)
        write_binary(folder / "weights.bin", w)
        (folder / "methods.txt").write_text(" ".join(METAP) + "\n")
    cases_file = directory / "combine" / "cases.txt"
    cases_file.write_text("".join(f"case{c}\n" for c in range(len(cases))))
    write_binary(directory / "probe.bin", probe_p_values())
    (directory / "pi0").mkdir()
    runs = []
    for s, p in enumerate(pi0.values()):
        write_binary(directory / "pi0" / f"set{s}.bin", p)
        runs += [
            f"set{s}.bin\t{json.dumps(options)}" for options in PI0_OPTIONS.values()
        ]
    (directory / "pi0" / "runs.txt").write_text("\n".join(runs) + "\n")
    lines = []
    columns = {"log2_fold_change": "lfc", "standard_error": "se", "p_value": "p"}
    columns["adjusted_p_value"] = "adj"
    for name, s, result in analyses:
        group = result.settings["group"]
        table = StudyResults(collection, results).table(group, "stim_vs_ctrl")
        k, n = collection.n_studies, collection.n_features
        ids = result.features["feature_id"].tolist()
        keep = np.isin(table["feature_id"][:n].tolist(), ids)
        tested = np.array(table["status"].tolist()).reshape(k, n) == "tested"
        folder = directory / "replication" / name
        folder.mkdir(parents=True)
        (folder / "shape.txt").write_text(f"{k}\n{int(keep.sum())}\n")
        for column, file in columns.items():
            values = np.where(tested, np.asarray(table[column]).reshape(k, n), np.nan)
            write_binary(folder / f"{file}.bin", values[:, keep].T)
        write_binary(folder / "n_samples.bin", result.studies["n_samples"])
        method, conjunction, u, alpha, meta, options = SETTINGS[s]
        options = DEFAULT_PI0 if options is None else options
        spec = {"name": name, "method": method, "conjunction": conjunction}
        spec |= {"at_least": u, "alpha": alpha, "meta": meta, "pi0": options}
        lines.append(json.dumps(spec))
    (directory / "replication" / "analyses.txt").write_text("\n".join(lines) + "\n")
    rscript = os.environ.get("RSCRIPT", "Rscript")
    metafor = os.environ.get("METAFOR_RSCRIPT", rscript)
    for program, text in ((rscript, R_META), (metafor, R_METAFOR)):
        (directory / "program.R").write_text(text)
        subprocess.run(
            [program, "--vanilla", str(directory / "program.R"), str(directory)],
            check=True,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY)),
        )

    def read(path, rows):
        return np.fromfile(path).reshape(-1, rows).T

    shapes = {"genes": 12, "pairs": 6, "found_in": 5, "heterogeneity": 4}
    replication = {
        name: {
            part: read(directory / "replication" / name / f"{part}.bin", rows)
            for part, rows in shapes.items()
        }
        for name, _, _ in analyses
    }
    rows = 3 * len(METAP) + 18
    versions = json.loads((directory / "versions.json").read_text())
    return {
        "scipy": dict(np.load(directory / "reference.npz")),
        "metap": [
            read(directory / "combine" / f"case{c}" / "metap.bin", rows)
            for c in range(len(cases))
        ],
        "qvalue": json.loads((directory / "pi0" / "qvalue.json").read_text()),
        "replication": replication,
        "probes": json.loads((directory / "probes.json").read_text()),
        "versions": [f"scipy {versions['scipy']}", f"mpmath {versions['mpmath']}"]
        + (directory / "versions_meta.txt").read_text().split("\n")[:4]
        + (directory / "versions_metafor.txt").read_text().split("\n")[:1],
    }


def difference(ours, reference, p_value=False):
    """The largest |ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|); equal values, NaN included, differ by 0."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    same = (ours == reference) | (np.isnan(ours) & np.isnan(reference))
    with np.errstate(invalid="ignore", divide="ignore"):
        if p_value:
            scale = np.maximum(reference, 1e-300)
            scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
        else:
            scale = np.maximum(np.abs(reference), 1.0)
        scaled = np.where(same, 0.0, np.abs(ours - reference) / scale)
    return float(np.max(np.where(np.isnan(scaled), np.inf, scaled), initial=0.0))


def report(check, reference, value, tolerance, metric="max scaled difference"):
    result = "pass" if value <= tolerance else "FAIL"
    return (check, reference, metric, f"{value:.2g}", f"{tolerance:g}", result)


def behaviour(check, reference, text, result="deliberate difference"):
    return (check, reference, "behaviour", text, "n/a", result)


def combined(p, signs, w, method, rank=None, directions=False):
    """CytePool's p-values and statistics, as rows; Wilkinson's method without a rank
    is metap's maximump, r = k, of the features of two studies or more."""
    from cytepool.stats.combined_p_values import combine_p_values

    if method == "wilkinson" and rank is None:
        k = np.count_nonzero(~np.isnan(p), axis=0)
        values = np.full((2, p.shape[1]), np.nan)
        for size in np.unique(k[k >= 2]):
            cut = k == size
            values[:, cut] = combined(p[:, cut], signs[:, cut], w, method, int(size), directions)  # fmt: skip
        return values
    r = combine_p_values(
        p,
        method=method,
        weights=w if method == "stouffer" else None,
        directions=signs if directions else None,
        rank=rank,
    )
    return np.array([r.p_value, r.statistic])


def check_combinations(cases, extreme, reference):
    from cytepool.stats.combined_p_values import partial_conjunction

    rows = []
    for name, (p, signs, w) in cases.items():
        k = np.count_nonzero(~np.isnan(p), axis=0)
        for method in SCIPY:
            p_value, statistic = combined(p, signs, w, method)
            if method == "pearson":  # scipy's statistic is -X
                statistic = -statistic
            if method == "mudholkar_george":  # and the sum of the logits
                statistic = (
                    statistic * np.pi / np.sqrt(3 / k * (5 * k + 4) / (5 * k + 2))
                )
            theirs = reference["scipy"][f"{name}/{method}"]
            value = difference(statistic, theirs[0])
            value = max(value, difference(p_value, theirs[1], True))
            rows.append(report(f"{name}: {method}", "combine_pvalues", value, EXACT))
    methods = "Fisher, Stouffer, Tippett, Wilkinson, the maximum, logit, Simes"
    for c, (name, (p, signs, w)) in enumerate(cases.items()):
        theirs = reference["metap"][c]
        several = np.count_nonzero(~np.isnan(p), axis=0) >= 2
        # two2one rounds 1 - p / 2: features whose p-values are all at least 1e-5.
        large = several & (np.nanmin(np.where(np.isnan(p), 1.0, p), axis=0) >= 1e-5)
        worst = [0.0, 0.0]
        for m, (metap, method) in enumerate(METAP.items()):
            rank = 2 if metap == "wilkinson" else None
            ours = combined(p, signs, w, method, rank)
            signed = combined(p, signs, w, method, rank, directions=True)
            value = difference(ours[1][several], theirs[3 * m][several])
            value = max(value, difference(ours[0][several], theirs[3 * m + 1][several], True))  # fmt: skip
            worst[0] = max(worst[0], value)
            value = difference(signed[0][large], theirs[3 * m + 2][large], True)
            worst[1] = max(worst[1], value)
        rows.append(report(f"{name}: {methods}", "metap, p.adjust", worst[0], EXACT))
        check = (
            f"{name}: the same with directions, two-sided ({int(large.sum())} features)"
        )
        rows.append(report(check, "metap, two2one", worst[1], EXACT))
        worst = [0.0, 0.0]
        for u in (1, 2, 3):
            for j, method in enumerate(CONJUNCTIONS):
                row = 3 * len(METAP) + 2 * (3 * (u - 1) + j)
                for d, use in ((0, several), (1, large)):
                    r = partial_conjunction(
                        p, u, method=method, directions=signs if d else None
                    )
                    value = difference(r.p_value[use], theirs[row + d][use], True)
                    worst[d] = max(worst[d], value)
        check = f"{name}: partial conjunctions, u = 1 to 3, by Simes, Fisher, Stouffer"
        rows.append(report(check, "definition, metap, p.adjust", worst[0], EXACT))
        check = f"{name}: the same with directions"
        rows.append(report(check, "definition, metap, two2one", worst[1], EXACT))
    p, signs, w = extreme
    # Half of a subnormal p-value loses digits: features with one are left out.
    normal = np.nanmin(p, axis=0) >= SMALLEST_NORMAL
    worst = 0.0
    for method in EXTREME:
        theirs = reference["scipy"][f"extreme/{method}"]
        rank = 2 if method == "wilkinson" else None
        ours = combined(p, signs, w, method, rank)
        signed = combined(p, signs, w, method, rank, directions=True)
        worst = max(worst, difference(ours[0], theirs[0], True))
        worst = max(worst, difference(ours[1], theirs[1]))
        worst = max(worst, difference(signed[0][normal], theirs[2][normal], True))
        worst = max(worst, difference(signed[1][normal], theirs[3][normal]))
    check = "13 features of p-values from 5e-324 to 1 - 1e-15, every method, with and without directions"  # fmt: skip
    rows.append(report(check, "mpmath, 400 digits", worst, EXACT))
    probes = reference["probes"]
    zero = probes["metap_zero"]
    ours = combined(np.array([0.0, 0.2, 0.5]), None, None, "fisher")[0]
    text = f"metap: {zero['value']:.4g} without it, '{zero['warning']}'; CytePool: {ours:g}"
    rows.append(behaviour("a p-value of 0, Fisher", "sumlog", text))
    ours = combined(np.array([0.2]), None, None, "fisher")[0]
    text = f"metap: '{probes['metap_one_study']['error']}'; CytePool: {ours:g}"
    rows.append(behaviour("one study, Fisher", "sumlog", text))
    return rows


def check_pi0(pi0, reference):
    from cytepool.stats.multiple_testing import proportion_of_true_nulls as estimate

    rows = []
    for label, options in PI0_OPTIONS.items():
        worst, stops = 0.0, 0
        for s, p in enumerate(pi0.values()):
            theirs = reference["qvalue"][f"set{s}.bin\t{json.dumps(options)}"]
            if "error" in theirs:
                stops += 1
                continue
            ours = estimate(p, **{NAMES[key]: value for key, value in options.items()})
            worst = max(worst, difference(ours.pi0, theirs["pi0"]))
            theirs_lambda = np.atleast_1d(theirs["pi0_lambda"])
            worst = max(worst, difference(ours.pi0_lambda, theirs_lambda))
            if not np.array_equal(ours.lambdas, np.atleast_1d(theirs["lambda"])):
                worst = np.inf
            smooth = theirs["pi0_smooth"]
            if (ours.pi0_smooth is None) != (smooth is None):
                worst = np.inf
            elif smooth is not None:
                worst = max(worst, difference(ours.pi0_smooth, smooth))
        check = f"pi0, {label}: {len(pi0) - stops} sets of p-values"
        check += f" (qvalue stops for {stops})" if stops else ""
        rows.append(report(check, "pi0est", worst, EXACT))
    probes, u = reference["probes"], probe_p_values()
    for label, key, ours in (
        ("no p-value at the largest lambdas", "zero_count_smoother", estimate(u * 0.9)),
        ("the same, bootstrap", "zero_count_bootstrap", estimate(u * 0.9, method="bootstrap")),
        ("an estimate of at most 0", "estimate_zero", estimate(u * 0.4, lambdas=0.5)),
    ):  # fmt: skip
        text = f"qvalue: '{probes[key]['error']}'; CytePool: {ours.pi0:.4g}"
        rows.append(behaviour(f"pi0, {label}", "pi0est", text))
    theirs = probes["missing_p"]["value"]
    value = difference(estimate(np.append(u**2, np.nan)).pi0, theirs)
    check = "pi0 with a missing p-value, which both leave out"
    rows.append(report(check, "pi0est", value, EXACT))
    for label, key, options in (
        (
            "smooth.log.pi0 where no p-value reaches a lambda",
            "zero_count_log",
            {"smooth_log": True},
        ),  # fmt: skip
        ("two values of lambda", "two_lambdas", {"lambdas": [0.2, 0.5]}),
        ("lambda = 1", "lambda_one", {"lambdas": 1.0}),
    ):
        try:
            estimate(u * 0.9 if key == "zero_count_log" else u, **options)
            refused = False
        except ValueError:
            refused = True
        both = refused and "error" in probes[key]
        result = "pass" if both else "FAIL"
        rows.append(behaviour(f"pi0, {label}", "pi0est", "both refuse", result))
    ours = estimate(u**1.3, lambdas=np.arange(96) / 100).pi0
    theirs = probes["lambdas_96"]["value"]
    text = (
        f"relative difference {abs(ours - theirs) / theirs:.2g}; knots at every lambda"
    )
    rows.append(behaviour("pi0, 96 values of lambda", "pi0est", text))
    return rows


def check_replication(analyses, reference):
    worst = dict.fromkeys(("genes", "reml", "dl", "pairs", "found_in"), 0.0)
    worst["default"] = 0.0
    compared = stops = by_default = 0
    counts = ("n_studies", "n_up", "n_down", "n_found", "n_found_up", "n_found_down")
    counts += ("sign",)
    values = ("sign_agreement", "p_value", "replication_p_value", "adjusted_p_value")
    values += ("replication_adjusted_p_value",)
    heterogeneity = ("q_statistic", "q_p_value", "i_squared", "tau_squared")
    for name, s, result in analyses:
        theirs = reference["replication"][name]
        genes, g = result.features, theirs["genes"]
        if any(not np.array_equal(genes[c], g[i]) for i, c in enumerate(counts)):
            worst["genes"] = np.inf
        for i, column in enumerate(values, len(counts)):
            worst["genes"] = max(worst["genes"], difference(genes[column], g[i], i > 7))
        meta = result.settings["meta_method"]
        for i, column in enumerate(heterogeneity):
            value = difference(genes[column], theirs["heterogeneity"][i], i == 1)
            worst[meta] = max(worst[meta], value)
        pairs, r = result.study_pairs, theirs["pairs"]
        same = np.array_equal(pairs["n_features_found"], r[2])
        if not (same and np.array_equal(pairs["n_features_found_in_both"], r[3])):
            worst["pairs"] = np.inf
        known = ~np.isnan(r[5])
        worst["pairs"] = max(worst["pairs"], difference(pairs["sign_agreement"], r[4]))
        worst["pairs"] = max(
            worst["pairs"], difference(pairs["pi1"][known], r[5][known])
        )
        if SETTINGS[s][-1] is None:  # pi1 by default, against pi0est(lambda = 0.5)
            worst["default"] = max(
                worst["default"], difference(pairs["pi1"][known], r[5][known])
            )
            by_default += int(known.sum())
        compared += int(known.sum())
        stops += int(np.count_nonzero(~known & ~np.isnan(pairs["pi1"])))
        f, found_in = theirs["found_in"], result.found_in
        same = np.array_equal(found_in["n_features"], f[1])
        if not (same and np.array_equal(found_in["n_features_same_direction"], f[2])):
            worst["found_in"] = np.inf
    n = len(analyses)
    reml = sum(result.settings["meta_method"] == "reml" for _, _, result in analyses)
    return [
        report(f"{n} replications, genes: counts, directions, sign agreement, combined and partial conjunction p-values, adjusted", "metap, p.adjust", worst["genes"], EXACT),
        report(f"{reml} replications with REML: Q, its p-value, I^2 and tau^2", "rma.uni", worst["reml"], CLOSE),
        report(f"{n - reml} replications with DerSimonian-Laird: the same", "rma.uni", worst["dl"], EXACT),
        report(f"{n} replications, pairs of studies: genes found, sign agreement, {compared} values of pi1", "pi0est", worst["pairs"], EXACT),
        report(f"pi1 by default (null_proportion left out), {by_default} values: pi0 at lambda = 0.5", "pi0est(lambda = 0.5)", worst["default"], EXACT),
        report(f"{n} replications: genes found in at least k studies, and in one direction", "R", worst["found_in"], 0.0, "mismatches"),
        behaviour(f"pi1 of {stops} pairs where qvalue stops: no p-value at its largest lambda, or an estimate of 0", "pi0est", "qvalue: an error; CytePool: an estimate"),
    ]  # fmt: skip


def check_figures(analyses, reference):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    from cytepool.plotting import plot_replicated_genes, plot_study_agreement

    worst = [0.0, 0.0]
    for name, _, result in analyses:
        theirs = reference["replication"][name]
        k = int(result.studies["analysed"].sum())
        expected = np.full((k, k), np.nan)
        pairs = theirs["pairs"]
        expected[pairs[0].astype(int) - 1, pairs[1].astype(int) - 1] = pairs[5]
        image = plot_study_agreement(result).get_images()[0]
        drawn = np.ma.filled(image.get_array().astype(float), np.nan)
        known = ~np.isnan(expected)
        worst[0] = max(worst[0], difference(drawn[known], expected[known]))
        bars = plot_replicated_genes(result).patches
        same = [b.get_height() for b in bars if b.get_gid() == "same direction"]
        both = [b.get_height() for b in bars if b.get_gid() == "both directions"]
        f = theirs["found_in"]
        worst[1] += np.count_nonzero(same != f[3]) + np.count_nonzero(both != f[4])
        pyplot.close("all")
    return [
        report("plot_study_agreement: pi1 of every pair, as drawn", "pi0est", worst[0], EXACT),
        report("plot_replicated_genes: genes found in exactly k studies, in one direction or both, as drawn", "R", worst[1], 0.0, "mismatches"),
    ]  # fmt: skip


def write_fixtures(cases, extreme, pi0, reference, small):
    """Small inputs and reference outputs for the tests."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    p, signs, w = cases["5 studies"]
    keep = np.arange(40)
    arrays = {"p": p[:, keep], "signs": signs[:, keep], "weights": w}
    arrays["metap"] = reference["metap"][list(cases).index("5 studies")][:, keep]
    for method in SCIPY:
        arrays[f"scipy/{method}"] = reference["scipy"][f"5 studies/{method}"][:, keep]
    parts = ("extreme/p", "extreme/signs", "extreme/weights")
    arrays |= dict(zip(parts, extreme, strict=True))
    for method in EXTREME:
        arrays[f"extreme/{method}"] = reference["scipy"][f"extreme/{method}"]
    np.savez_compressed(FIXTURES / "combined.npz", **arrays)
    arrays = {}
    for s, values in enumerate(pi0.values()):
        if values.size > 1000:
            continue
        arrays[f"set{s}/p"] = values
        for label, options in PI0_OPTIONS.items():
            theirs = reference["qvalue"][f"set{s}.bin\t{json.dumps(options)}"]
            if "error" not in theirs:
                arrays[f"set{s}/{label}/pi0"] = np.array(theirs["pi0"])
                if theirs["pi0_smooth"] is not None:
                    arrays[f"set{s}/{label}/pi0_smooth"] = np.array(
                        theirs["pi0_smooth"]
                    )
    np.savez_compressed(FIXTURES / "pi0.npz", **arrays)
    cubes, analyses, replicated = small
    arrays = {}
    for study, cube in cubes.items():
        arrays[f"{study}/counts"] = cube.layer("counts")
        arrays[f"{study}/n_cells"] = cube.n_cells
        arrays[f"{study}/feature_id"] = np.array(cube.features["feature_id"].tolist())
        for column in ("sample_id", "donor", "condition"):
            arrays[f"{study}/{column}"] = np.array(cube.samples[column].tolist())
    for name, _, _ in analyses:
        for part, values in replicated[name].items():
            arrays[f"{name}/{part}"] = values
    np.savez_compressed(FIXTURES / "replication.npz", **arrays)
    print(f"wrote combined.npz, pi0.npz and replication.npz to {FIXTURES}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.reference is not None:
        reference_main(arguments.reference)
        return 0
    rng = np.random.default_rng(SEED)
    cases, extreme, pi0 = p_value_cases(rng), extreme_case(), pi0_sets(rng)
    _, collection, results = cohorts(np.random.default_rng(SEED + 1), 700)
    analyses = replications(collection, results)
    inputs = (cases, extreme, pi0, analyses, collection, results)
    with tempfile.TemporaryDirectory() as temporary:
        reference = run_references(Path(temporary), *inputs)
    print("Reference: " + ", ".join(reference["versions"]))
    table = check_combinations(cases, extreme, reference) + check_pi0(pi0, reference)
    table += check_replication(analyses, reference) + check_figures(analyses, reference)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join("| " + " | ".join(map(str, line)) + " |" for line in table))
    failures = sum(line[-1] == "FAIL" for line in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and not failures:
        cubes, collection, results = cohorts(np.random.default_rng(SEED + 2), 160, (4, 3, 5))  # fmt: skip
        analyses = replications(collection, results, SETTINGS[:2])[:2]
        inputs = (cases, extreme, pi0, analyses, collection, results)
        with tempfile.TemporaryDirectory() as temporary:
            small = run_references(Path(temporary), *inputs)["replication"]
        write_fixtures(cases, extreme, pi0, reference, (cubes, analyses, small))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
