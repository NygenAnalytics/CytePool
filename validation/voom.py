"""Check voom precision weights against limma.

Compares ``cytepool.stats.voom.voom`` with limma's ``voom`` on negative binomial counts
simulated from fixed seeds: 2 to 20,000 genes and 4 to 40 samples. The cases cover two
groups, four groups with a covariate, group means, a covariate without an intercept,
one residual degree of freedom, normalization factors (edgeR's ``calcNormFactors`` in
R, CytePool's ``tmm_factors`` here) with and without given library sizes, genes with
zero counts, two very small libraries, libraries from 1e3 to 1e8 reads, an
intercept-only design, genes with the same counts and blocks of genes with a single
read (ties in the trend), rarefied counts, estimated (non-integer) counts, spans of 0.3
and 1, a design without replication and two genes, each with a fixed span (0.5 unless
named; limma's ``adaptive.span = FALSE``). Seven more choose the span from the number
of genes (``adaptive.span = TRUE``, limma's default since 3.66 and CytePool's): 2, 30,
51 and 120 genes, about a fifth of the last three without counts, the group means and
four-group cases, and the design without replication. Each design is built from the
same sample table by R's ``model.matrix`` and by CytePool's ``Design``, and its
contrasts by ``makeContrasts`` and by the design's named contrasts.
``choose_lowess_span`` is compared with ``chooseLowessSpan`` over 0 to 1e9 points with
limma's settings, and over other settings.

For each case the table reports the largest difference in: the log2 CPM ``E`` and the
library sizes (with the normalization factors); the span, where it is chosen; the
weights; the trend (the points and LOWESS line voom saves with ``save.plot=TRUE``); and
the moderated t-statistics and p-values after ``lmFit`` with the voom weights,
``contrasts.fit`` and ``eBayes``, against ``fit_linear_model``,
``contrasts_fit(method="limma")`` and ``moderated_statistics``. Differences are
relative, except for: E, the trend's x and t, scaled by max(|reference|, 1); the
trend's y, scaled by max(|reference|, median of the reference), because near zero both
are rounding noise; and p-values, scaled by max(p, 1e-300) max(1, |log p|) as in
``empirical_bayes.py``. The cases without replication check that both warn.
Deliberate deviations, where limma returns values without meaning or fails, are listed
as such, with what each program does.

Tiers: exact (1e-10), except rarefied counts (close, 1e-6). E and the spans are closed
forms and agree to the last bit, and so does the trend's x. The weights depend on lmFit's
residual standard deviations, which a different QR reproduces up to rounding: about
1e-16 times |E| over the residual standard deviation, relative. LOWESS and the
interpolation are continuous in them, so the weights agree to about 1e-13, and to
about 1e-12 for a gene that holds almost all the counts (two genes). Rarefied counts
have equal library sizes, so a gene with the same count in every sample has a constant
log2 CPM and a residual standard deviation that is pure rounding noise, about 1e-15 in
either program. These points of the trend differ by up to about 1e-7 of the typical
square-root standard deviation, which moves the weights by about 2e-8 (3e-10 in the
fixture). E and the library sizes stay in the exact tier there too.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages limma and edgeR. Set ``RSCRIPT`` to
    the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/voom.py
    PYTHONPATH=. python validation/voom.py --write-fixtures
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from cytepool.design import Design
from cytepool.normalization import tmm_factors
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.smoothing import choose_lowess_span
from cytepool.stats.voom import voom

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "voom"
EXACT = 1e-10
CLOSE = 1e-6
# Outputs compared by group, and the floor of the scale of their differences.
TREND = {"xy.x": 1.0, "xy.y": "median", "line.x": 1.0, "line.y": "median"}
GROUPS = {
    "E, library sizes": {"E": 1.0, "lib.size": 0.0, "norm.factors": 0.0},
    "chosen span": {"span": 0.0},
    "weights": {"weights": 0.0},
    "trend points, LOWESS line": TREND,
    "moderated t, p": {"t": 1.0, "p.value": 1e-300},
}
# Families of the span rule where CytePool deliberately differs from limma, with
# what CytePool returns for each setting (None: it raises an error).
RULE_DEVIATIONS = {
    "0 points, where limma's formula gives NaN": 1.0,
    "outside the accepted ranges": None,
}
# Counts of 8 genes in 4 samples (two groups of two) whose LOWESS trend is negative
# at its upper end.
NEGATIVE_TREND = [
    [2, 101, 23, 131, 509, 87, 230, 150],
    [0, 253, 77, 60, 1306, 46, 51, 240],
    [1, 156, 169, 74, 3276, 19, 106, 265],
    [1, 17, 190, 30, 3434, 124, 162, 573],
]

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
has <- function(dir, name) file.exists(file.path(dir, name))
rd <- function(dir, name) {
  path <- file.path(dir, paste0(name, ".bin"))
  readBin(path, "double", n = file.size(path) / 8, size = 8)
}
wr <- function(dir, name, x) writeBin(as.double(x), file.path(dir, paste0("out_", name, ".bin")), size = 8)
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  samples <- read.delim(file.path(dir, "samples.tsv"), stringsAsFactors = TRUE)
  X <- model.matrix(as.formula(readLines(file.path(dir, "formula.txt"))), samples)
  counts <- matrix(rd(dir, "counts"), ncol = nrow(X))
  lib <- if (has(dir, "lib.bin")) rd(dir, "lib") else NULL
  # NA: choose the span from the number of genes, which ignores span.
  span <- scan(file.path(dir, "span.txt"), quiet = TRUE)
  caught <- character(0)
  keep <- function(w) { caught <<- c(caught, conditionMessage(w)); invokeRestart("muffleWarning") }
  failed <- function(e) { writeLines(conditionMessage(e), file.path(dir, "out_error.txt")); NULL }
  # lmFit prints the coefficients it cannot estimate; the output is not needed.
  printed <- capture.output(v <- tryCatch(suppressMessages(withCallingHandlers(
    if (has(dir, "tmm.txt")) {
      y <- calcNormFactors(DGEList(counts, lib.size = lib))
      wr(dir, "norm.factors", y$samples$norm.factors)
      voom(y, X, span = span, adaptive.span = is.na(span), save.plot = TRUE)
    } else {
      voom(counts, X, lib.size = lib, span = span, adaptive.span = is.na(span),
           save.plot = TRUE)
    }, warning = keep)), error = failed))
  writeLines(caught, file.path(dir, "out_warnings.txt"))
  if (is.null(v) || has(dir, "deviation.txt")) next
  wr(dir, "E", v$E); wr(dir, "weights", v$weights); wr(dir, "lib.size", v$targets$lib.size)
  if (!is.null(v$span)) wr(dir, "span", v$span)
  if (!is.null(v$voom.xy)) {
    wr(dir, "xy.x", v$voom.xy$x); wr(dir, "xy.y", v$voom.xy$y)
    wr(dir, "line.x", v$voom.line$x); wr(dir, "line.y", v$voom.line$y)
  }
  if (nrow(X) > ncol(X)) {
    fit <- lmFit(v, X)
    if (has(dir, "contrasts.tsv")) {
      cs <- read.delim(file.path(dir, "contrasts.tsv"), stringsAsFactors = FALSE)
      # makeContrasts renames (Intercept) in its row names, with a warning.
      C <- suppressWarnings(makeContrasts(contrasts = cs$expression, levels = X))
      fit <- contrasts.fit(fit, C)
    }
    # eBayes warns about the zero variances of the rarefied case.
    eb <- suppressWarnings(eBayes(fit))
    wr(dir, "t", eb$t); wr(dir, "p.value", eb$p.value)
  }
}
# The span rule on rows of n, small.n, min.span and power.
rule <- matrix(rd(root, "rule"), ncol = 4)
spans <- mapply(chooseLowessSpan, rule[, 1], rule[, 2], rule[, 3], rule[, 4])
writeBin(as.double(spans), file.path(root, "out_rule.bin"), size = 8)
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR"))), file.path(root, "versions.txt"))
"""


def simulate(rng, n_genes, groups, sizes, *, zero=0.0, up=0.0):
    """Samples x genes negative binomial counts with group effects.

    A fraction ``zero`` of the genes has no counts, and a fraction ``up`` is 8-fold
    higher in the second group, which changes the library composition so that TMM
    factors differ from one.
    """
    levels, index = np.unique(groups, return_inverse=True)
    proportion = np.exp(rng.normal(0.0, 2.0, n_genes))
    proportion /= proportion.sum()
    lfc = np.zeros((levels.size, n_genes))
    changed = rng.random(n_genes) < 0.1
    lfc[1:, changed] = rng.normal(0.0, 1.5, (levels.size - 1, changed.sum()))
    if up:
        lfc[1, rng.random(n_genes) < up] = 3.0
    mean = sizes[:, np.newaxis] * proportion * 2.0 ** lfc[index]
    size = 1.0 / (0.05 * np.exp(rng.normal(0.0, 0.5, n_genes)))
    counts = rng.negative_binomial(size, size / (size + mean)).astype(float)
    counts[:, rng.random(n_genes) < zero] = 0.0
    return counts


def make_cases(rng, small):
    """Cases by name; ``small`` gives the sizes of the test fixtures."""
    genes, fewer, most = (200, 100, 200) if small else (10_000, 2_000, 20_000)
    cases = {}

    def add(name, samples, formula, counts, **settings):
        cases[name] = {"samples": samples, "formula": formula, "counts": counts}
        cases[name] |= {"span": 0.5, "tmm": False} | settings

    two = {"group": np.repeat(["A", "B"], 3)}
    counts = simulate(rng, genes, two["group"], rng.uniform(1e6, 3e6, 6), zero=0.05)
    add("two_groups", two, "~ group", counts)
    add("span_0.3", two, "~ group", counts, span=0.3)
    add("span_1", two, "~ group", counts, span=1.0)
    lib = counts.sum(axis=1) * rng.uniform(0.7, 1.4, 6)
    add("library_sizes_given", two, "~ group", counts, lib=lib)
    counts = simulate(rng, genes, two["group"], rng.uniform(1e6, 3e6, 6), up=0.1)
    add("two_groups_tmm", two, "~ group", counts, tmm=True)
    lib = counts.sum(axis=1) * rng.uniform(0.7, 1.4, 6)
    add("library_sizes_and_tmm", two, "~ group", counts, tmm=True, lib=lib)
    n = 12 if small else 40
    samples = {"group": np.repeat(list("ABCD"), n // 4)}
    samples["age"] = rng.uniform(20.0, 70.0, n).round(1)
    sizes = rng.uniform(5e5, 5e6, n)
    counts = simulate(rng, most, samples["group"], sizes, zero=0.02, up=0.05)
    contrasts = {"BvsA": "groupB", "CvsB": "groupC - groupB", "DvsC": "groupD - groupC"}
    add("groups_covariate_tmm", samples, "~ group + age", counts, tmm=True)
    cases["groups_covariate_tmm"]["contrasts"] = contrasts
    three = {"group": np.repeat(list("ABC"), 3 if small else 4)}
    sizes = rng.uniform(1e6, 2e6, 9 if small else 12)
    counts = simulate(rng, genes, three["group"], sizes)
    add("group_means_tmm", three, "~ 0 + group", counts, tmm=True)
    contrasts = {"BvsA": "groupB - groupA", "CvsB": "groupC - groupB"}
    cases["group_means_tmm"]["contrasts"] = contrasts
    ages = {"age": rng.uniform(20.0, 70.0, 8).round(1)}
    counts = simulate(rng, fewer, np.repeat(["A"], 8), rng.uniform(1e6, 3e6, 8))
    add("covariate_only", ages, "~ 0 + age", counts)
    unequal = {"group": np.array(["A", "A", "B", "C"])}
    counts = simulate(rng, fewer, unequal["group"], rng.uniform(1e6, 3e6, 4))
    add("one_residual_df", unequal, "~ group", counts)
    four = {"group": np.repeat(list("ABC"), 4)}
    sizes = rng.uniform(1e6, 2e6, 12)
    sizes[[1, 6]] = [5e3, 2e4]
    add("small_libraries", four, "~ group", simulate(rng, genes, four["group"], sizes))
    alternating = {"group": np.tile(["A", "B"], 4)}
    sizes = np.geomspace(1e3, 1e8, 8)
    counts = simulate(rng, genes, alternating["group"], sizes, zero=0.05)
    add("extreme_libraries", alternating, "~ group", counts)
    one = {"group": np.repeat(["A"], 8)}
    counts = simulate(rng, fewer, one["group"], rng.uniform(1e6, 3e6, 8))
    add("intercept_only", one, "~ 1", counts)
    # About one count per gene: many genes share a count profile, and so a point of
    # the trend.
    counts = simulate(rng, genes, two["group"], np.full(6, 1.0 * genes))
    add("ties", two, "~ group", counts)
    # A single read in one sample or another gives the same x in exact arithmetic:
    # a block of ties wider than the LOWESS window. Its ties, and so the trend,
    # depend on the last bit of x. The seeds are chosen to expose that, as about a
    # quarter of seeds do: with means accumulated in double precision (numpy's)
    # rather than rounded once (R's rowMeans), the weights would move by 12% and 6%.
    expressed, single = (200, 150) if small else (2_000, 1_500)
    tied = np.random.default_rng(4 if small else 7)
    counts = np.zeros((6, expressed + single))
    counts[:, :expressed] = simulate(tied, expressed, two["group"], np.full(6, 3e4))
    counts[tied.integers(0, 6, single), expressed + np.arange(single)] = 1.0
    add("tie_blocks", two, "~ group", counts, span=0.3)
    # Rarefied to equal depth, plus genes with the same count in every sample: equal
    # library sizes, so those genes have a constant log2 CPM.
    counts = simulate(rng, genes, two["group"], rng.uniform(1e6, 3e6, 6))
    constant = rng.random(genes) < 0.05
    varying = counts[:, ~constant].astype(int)
    depth = int(varying.sum(axis=1).min() // 2)
    counts[:, ~constant] = [rng.multivariate_hypergeometric(r, depth) for r in varying]
    counts[:, constant] = rng.integers(1, 50, constant.sum())
    add("rarefied", two, "~ group", counts, tolerance=CLOSE)
    counts = simulate(rng, fewer, two["group"], rng.uniform(1e6, 3e6, 6))
    add("non_integer", two, "~ group", counts * rng.uniform(0.5, 1.5, counts.shape))
    saturated = {"sample": np.array([f"S{i}" for i in range(4)])}
    counts = simulate(rng, fewer, saturated["sample"], rng.uniform(1e6, 3e6, 4))
    add("no_replication", saturated, "~ sample", counts)
    counts = simulate(rng, 2, two["group"], np.full(6, 1e3)) + 5.0
    add("two_genes", two, "~ group", counts)
    # Spans chosen from the number of genes (span None). limma counts the genes
    # without counts, which the trend leaves out: with 51 genes, each without counts
    # with probability 0.2, the span is below 1 although at most 50 enter the trend.
    for n_genes in (30, 51, 120):
        sizes = rng.uniform(1e6, 3e6, 6)
        counts = simulate(rng, n_genes, two["group"], sizes, zero=0.2)
        add(f"adaptive_{n_genes}_genes", two, "~ group", counts, span=None)
    for name in "two_genes group_means_tmm groups_covariate_tmm no_replication".split():
        cases[f"adaptive_{name}"] = cases[name] | {"span": None}
    return cases


def span_rule_settings():
    """Rows of n, small.n, min.span and power for the span rule, by family."""
    cubes = 50 * np.arange(2, 11) ** 3  # 50 / n is a perfect cube
    n = np.unique(np.r_[np.arange(201), cubes, np.geomspace(200, 1e9, 41).round()])
    sizes = (1, 2, 9, 10, 11, 100, 1e4, 1e7)
    other = itertools.product(sizes, (0, 1, 10), (0.05, 0.5, 1), (0, 0.5, 1))
    # 0 points: limma's formula takes 0 times infinity (min.span 1) or 0 / 0
    # (small.n 0), NaN, where its help page gives 1, the span up to small.n points.
    nan = [(0, 1, 1, 0.5), (0, 10, 1, 1), (0, 0, 0.3, 1 / 3), (0, 0, 0.5, 1)]
    outside = [(-1, 50, 0.3, 1 / 3), (100, 50, 0, 1 / 3), (10, 50, 1.5, 1 / 3)]
    outside += [(10, 50, 0.3, -1), (100, 50, 0.3, 2)]
    return {
        "limma's settings, 0 to 1e9 points": [(k, 50, 0.3, 1 / 3) for k in n],
        "small.n of 0 to 10, min.span of 0.05 to 1, power of 0 to 1": list(other),
        "0 points, where limma's formula gives NaN": nan,
        "outside the accepted ranges": outside,
    }


def deviation_cases(rng):
    """Inputs where CytePool deliberately differs from limma, with what each does."""
    two = {"group": np.repeat(["A", "B"], 3)}
    counts = simulate(rng, 500, two["group"], rng.uniform(1e6, 3e6, 6))
    empty = np.where(np.arange(6)[:, np.newaxis] == 2, 0.0, counts)
    negative = np.array(NEGATIVE_TREND, dtype=float)
    pairs = {"group": np.repeat(["A", "B"], 2)}
    tied = np.repeat([[10.0], [25.0], [14.0], [40.0], [22.0], [35.0]], 2, axis=1)
    lib = np.array([1e3, 1.2e3, 0.9e3, 1.5e3, 1.1e3, 1.3e3])
    batch = two | {"batch": np.repeat(["x", "y"], 3)}
    cases = {  # samples, formula, counts, settings; limma, CytePool (values or error)
        "sample without counts": (two, "~ group", empty, {}, "values", "library size is zero"),
        "trend negative": (pairs, "~ group", negative, {}, "values", "trend falls to"),
        "every x tied": (two, "~ group", tied, {"lib": lib}, "error", "values"),
        "aliased column": (batch, "~ group + batch", counts, {}, "values", "full column rank"),
        "span above 1": (two, "~ group", counts, {"span": 1.5}, "values", "span must be"),
    }  # fmt: skip
    return {
        name: {"samples": s, "formula": f, "counts": c, "span": 0.5, "tmm": False}
        | settings
        | {"deviation": (limma, ours)}
        for name, (s, f, c, settings, limma, ours) in cases.items()
    }


def run_limma(cases, rule):
    """limma's outputs for each case, its spans for the rows of the span rule, and
    the versions of R, limma and edgeR."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, case in cases.items():
            directory = root / names[name]
            directory.mkdir()
            columns = list(case["samples"])
            rows = zip(*(case["samples"][c] for c in columns), strict=True)
            lines = ["\t".join(columns)] + ["\t".join(map(str, row)) for row in rows]
            (directory / "samples.tsv").write_text("\n".join(lines) + "\n")
            (directory / "formula.txt").write_text(case["formula"] + "\n")
            span = "NA" if case["span"] is None else repr(case["span"])
            (directory / "span.txt").write_text(span + "\n")
            (directory / "counts.bin").write_bytes(case["counts"].tobytes())
            if "lib" in case:
                (directory / "lib.bin").write_bytes(case["lib"].tobytes())
            if case["tmm"]:
                (directory / "tmm.txt").write_text("1\n")
            if "deviation" in case:
                (directory / "deviation.txt").write_text("1\n")
            if case.get("contrasts"):
                pairs = [f"{k}\t{v}" for k, v in case["contrasts"].items()]
                text = "\n".join(["name\texpression", *pairs]) + "\n"
                (directory / "contrasts.tsv").write_text(text)
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "rule.bin").write_bytes(np.asarray(rule, dtype=float).tobytes("F"))
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        for name, case in cases.items():
            directory = root / names[name]
            case["limma"] = {
                path.stem[4:]: np.fromfile(path, dtype="<f8")
                for path in directory.glob("out_*.bin")
            }
            text = (directory / "out_warnings.txt").read_text().strip()
            case["limma warnings"] = text.splitlines()
            case["limma failed"] = (directory / "out_error.txt").exists()
        spans = np.fromfile(root / "out_rule.bin", dtype="<f8")
        return spans, (root / "versions.txt").read_text().split("\n")[:3]


def run_voom(case):
    """voom on a case, from the counts alone, and whether it warned."""
    design = Design(case["samples"], case["formula"], contrasts=case.get("contrasts"))
    lib = case.get("lib")
    factors = tmm_factors(case["counts"], library_sizes=lib) if case["tmm"] else None
    keywords = {"library_sizes": lib, "norm_factors": factors, "span": case["span"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = voom(case["counts"], design, **keywords)
    warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
    return design, factors, result, warned


def cytepool_results(case):
    """CytePool's outputs for a case, keyed and shaped as limma writes them."""
    design, factors, result, warned = run_voom(case)
    out = {"warned": warned, "E": result.log_cpm, "weights": result.weights}
    out["lib.size"] = result.effective_library_sizes
    if case["span"] is None:
        out["span"] = result.span  # NaN where limma returns no span
    if factors is not None:
        out["norm.factors"] = factors
    in_trend = ~np.isnan(result.sqrt_sd)
    if in_trend.any():
        x = result.mean_log_count[in_trend]
        order = np.argsort(x, kind="stable")
        out |= {"xy.x": x, "xy.y": result.sqrt_sd[in_trend], "line.x": x[order]}
        out["line.y"] = result.sqrt_sd_trend[in_trend][order]
        fit = fit_linear_model(result.log_cpm, design, weights=result.weights)
        if len(design.contrasts):
            fit = contrasts_fit(fit, design.contrasts, method="limma")
        moderated = moderated_statistics(fit)
        # R writes matrices column by column.
        out["t"] = moderated.t.T
        out["p.value"] = moderated.p_value.T
    return out


def difference(ours, reference, floor=0.0, log_scaled=False):
    """Largest |ours - reference| / scale; inf if the shapes or NaN differ.

    The scale is max(|reference|, floor), with "median" the median of |reference|,
    times max(1, |log scale|) if log_scaled.
    """
    ours = np.asarray(ours, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if floor == "median":
        floor = float(np.median(np.abs(reference)))
    if ours.shape != reference.shape or not np.array_equal(
        np.isnan(ours), np.isnan(reference)
    ):
        return np.inf
    finite = ~np.isnan(reference)
    if not finite.any():
        return 0.0
    scale = np.maximum(np.abs(reference[finite]), floor)
    scale = np.where(scale > 0, scale, 1.0)
    if log_scaled:
        scale *= np.maximum(1.0, np.abs(np.log(scale)))
    return float(np.max(np.abs(ours[finite] - reference[finite]) / scale))


def compare(cases, table):
    """Rows of (check, reference, metric, value, tolerance, result); counts bitwise
    agreement of E, the trend's x and the chosen span."""
    identical = dict.fromkeys(("E", "xy.x", "with trend", "span", "with span"), 0)
    for name, case in cases.items():
        if "deviation" in case:
            table.append(deviation_row(name, case))
            continue
        ours, limma = cytepool_results(case), case["limma"]
        identical["E"] += np.array_equal(ours["E"].reshape(-1), limma["E"])
        if "xy.x" in limma:
            identical["with trend"] += 1
            identical["xy.x"] += np.array_equal(ours.get("xy.x"), limma["xy.x"])
        if "span" in limma:
            identical["with span"] += 1
            identical["span"] += np.array_equal(ours["span"], limma["span"][0])
        for group, keys in GROUPS.items():
            # E, the library sizes and the span are closed forms: exact in every case.
            closed = group.startswith(("E", "chosen"))
            tolerance = EXACT if closed else case.get("tolerance", EXACT)
            values = [
                difference(ours.get(k), limma.get(k, [np.nan]), floor, k == "p.value")
                for k, floor in keys.items()
                if k in limma or k in ours
            ]
            if values:
                result = "pass" if max(values) <= tolerance else "FAIL"
                cells = (f"{name}: {group}", "voom" + _source(group))
                cells += ("max scaled difference", f"{max(values):.1e}")
                table.append((*cells, f"{tolerance:.0e}", result))
        if name.removesuffix("_fixture").endswith("no_replication"):
            agree = ours["warned"] and any(
                "no replication" in w for w in case["limma warnings"]
            )
            outcome = ("both warn", "pass") if agree else ("warnings differ", "FAIL")
            cells = (f"{name}: warning", "voom", "behaviour", outcome[0], "n/a")
            table.append((*cells, outcome[1]))
    return identical


def cytepool_span(n, small_n, min_span, power):
    """choose_lowess_span, or None where it refuses the arguments."""
    try:
        return choose_lowess_span(
            int(n), small_n=int(small_n), min_span=min_span, power=power
        )
    except ValueError:
        return None


def compare_span_rule(rule, spans, table):
    """Rows for the span rule. Returns the settings where CytePool agrees with
    limma, each followed by limma's span, and how many spans are limma's to the
    last bit."""
    agree, identical, start = [], 0, 0
    for family, rows in rule.items():
        reference, start = spans[start : start + len(rows)], start + len(rows)
        ours = [cytepool_span(*row) for row in rows]
        cells = (f"span rule: {family}", "chooseLowessSpan")
        if family in RULE_DEVIATIONS:
            expected, nan = RULE_DEVIATIONS[family], int(np.isnan(reference).sum())
            value = f"limma: {len(rows) - nan} spans, {nan} NaN; CytePool: "
            value += "errors" if expected is None else f"{expected:g}"
            deviates = expected is None or nan == len(rows)  # limma: NaN, not 1
            same = ours == [expected] * len(rows)
            result = "deviation" if same and deviates else "FAIL"
            table.append((*cells, "behaviour", value, "n/a", result))
            continue
        agree += [(*row, span) for row, span in zip(rows, reference, strict=True)]
        identical += int(np.sum(np.asarray(ours, dtype=float) == reference))
        value = difference(ours, reference)
        result = "pass" if value <= EXACT else "FAIL"
        cells += ("max relative difference", f"{value:.1e}", f"{EXACT:.0e}", result)
        table.append(cells)
    return np.asarray(agree, dtype=float), identical


def deviation_row(name, case):
    limma = "error" if case["limma failed"] else "values"
    try:
        run_voom(case)
        ours = "values"
    except ValueError as error:
        # The expected error, not any other one.
        expected_message = case["deviation"][1]
        ours = "error" if expected_message in str(error) else f"other error: {error}"
    expected = (
        case["deviation"][0],
        "values" if case["deviation"][1] == "values" else "error",
    )
    result = "deviation" if (limma, ours) == expected else "FAIL"
    value = f"limma: {limma}; CytePool: {ours}"
    return (f"deviation: {name}", "voom", "behaviour", value, "n/a", result)


def _source(group):
    return ", lmFit, contrasts.fit, eBayes" if group.startswith("moderated") else ""


def write_fixtures(cases, rule):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, case in cases.items():
        arrays = {"counts": case["counts"], "formula": np.array(case["formula"])}
        arrays |= {f"samples {k}": np.asarray(v) for k, v in case["samples"].items()}
        if case["span"] is not None:  # otherwise chosen from the number of genes
            arrays["span"] = np.array(case["span"])
        arrays["tmm"] = np.array(case["tmm"])
        arrays["tolerance"] = np.array(case.get("tolerance", EXACT))
        if "lib" in case:
            arrays["lib"] = case["lib"]
        if case.get("contrasts"):
            arrays["contrast names"] = np.array(list(case["contrasts"]))
            arrays["contrast expressions"] = np.array(list(case["contrasts"].values()))
        arrays |= {f"limma {k}": v for k, v in case["limma"].items()}
        arrays["limma warnings"] = np.array(case["limma warnings"], dtype=str)
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    # The settings of the span rule where CytePool agrees with limma, and its spans.
    path = FIXTURE_DIRECTORY / "span_rule.npz"
    np.savez_compressed(path, settings=rule[:, :4], limma=rule[:, 4])
    print(f"wrote {len(cases) + 1} fixtures to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(1), small=False)
    cases |= deviation_cases(np.random.default_rng(3))
    if arguments.write_fixtures:
        small = make_cases(np.random.default_rng(2), small=True)
        cases |= {f"{name}_fixture": case for name, case in small.items()}
    rule = span_rule_settings()
    rows = [row for rows in rule.values() for row in rows]
    spans, versions = run_limma(cases, rows)
    print("Reference: " + ", ".join(versions))
    table = []
    identical = compare(cases, table)
    rule_agree, rule_identical = compare_span_rule(rule, spans, table)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    compared = sum("deviation" not in case for case in cases.values())
    print(
        f"identical to limma's to the last bit: E in {identical['E']} of {compared} "
        f"cases, the trend's x in {identical['xy.x']} of {identical['with trend']}, "
        f"the chosen span in {identical['span']} of {identical['with span']}, and "
        f"the span rule in {rule_identical} of {len(rule_agree)} settings"
    )
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        small = {
            n.removesuffix("_fixture"): c for n, c in cases.items() if "_fixture" in n
        }
        write_fixtures(small, rule_agree)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
