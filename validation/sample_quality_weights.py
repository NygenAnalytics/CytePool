"""Check sample quality weights, voom with prior weights and voom with quality weights.

Compares ``sample_quality_weights`` with limma's ``arrayWeights``, ``voom(weights=...)``
with limma's ``voom(weights = ...)``, and ``voom_with_quality_weights`` with
``voomWithQualityWeights``, on data simulated from fixed seeds:

- Sample weights of normal log-expression values with one or two samples two to four
  times noisier, and of voom's log2 CPM with its weights: 4 to 12 samples, 2,000 or
  20,000 genes; two groups, an intercept, three groups and a covariate; REML and the
  gene-by-gene algorithm; a weight per sample, per group (``var.group``) or from a
  covariate (``var.design``); ``prior.n`` of 0, 10 and 50; observation weights;
  missing values, genes with one residual df or no residual variance; a criterion that
  rises, where limma's REML stops early; one gene; no replication.
- voom with sample, observation and feature weights, observation weights with zeros,
  and the span chosen from the number of genes.
- voom with quality weights: both algorithms, the span fixed and chosen, weights per
  group and from log library size, TMM factors with given library sizes and
  contrasts, two noisy pseudobulk samples among 20,000 genes, no replication.

Designs come from one sample table through R's ``model.matrix`` and CytePool's
``Design``, contrasts through ``makeContrasts`` and named contrasts. Each row gives the
largest difference in one output of a case: the sample weights; the log2 CPM ``E``;
the weights (limma's ``voom`` returns the trend's alone, which are multiplied by the
prior weights first); the trend voom saves with ``save.plot = TRUE``; the chosen span;
or the moderated t and p after ``lmFit``, ``contrasts.fit`` and ``eBayes`` (with its
default prior estimator, which CytePool's default follows) against ``fit_linear_model``,
``contrasts_fit(method="limma")`` and ``moderated_statistics``. Differences are
relative, except for E, the trend's x and t, scaled by max(|reference|, 1); the
trend's y, by max(|reference|, its median); and p, by max(p, 1e-300) max(1, |log p|).
Where limma reports the genes REML leaves out, an early stop of REML or no
replication, CytePool must report the same. Deliberate deviations are listed with what each program does.

Tiers: the sample weights are iterative, in the close tier (1e-6), as is everything
computed from them; CytePool repeats limma's iterations and stopping rule, so they
agree to rounding. voom with prior weights is exact (1e-10), as in ``voom.py``.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages limma and edgeR, and jsonlite. Set
    ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/sample_quality_weights.py
    PYTHONPATH=. python validation/sample_quality_weights.py --write-fixtures
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from cytepool import Design
from cytepool.normalization import tmm_factors
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.sample_weights import sample_quality_weights
from cytepool.stats.voom import voom, voom_with_quality_weights

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_quality_weights"
)
EXACT = 1e-10
CLOSE = 1e-6
R_METHODS = {"auto": "auto", "gene_by_gene": "genebygene", "reml": "reml"}
REFERENCES = {"weights": "arrayWeights", "voom": "voom"}
# Outputs compared by group, with the floor of the scale of their differences.
TREND = {"xy.x": 1.0, "xy.y": "median", "line.x": 1.0, "line.y": "median"}
GROUPS = {
    "sample weights": {"sample.weights": 0.0},
    "E": {"E": 1.0},
    "weights": {"weights": 0.0},
    "trend points, LOWESS line": TREND,
    "chosen span": {"span": 0.0},
    "moderated t, p": {"t": 1.0, "p.value": 1e-300},
}
EXACT_GROUPS = {"E", "chosen span"}  # closed forms in every case

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR); library(jsonlite)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(dir, name) {
  path <- file.path(dir, paste0(name, ".bin"))
  if (file.exists(path)) readBin(path, "double", n = file.size(path) / 8, size = 8)
}
wr <- function(dir, name, x) writeBin(as.double(x), file.path(dir, paste0("out_", name, ".bin")), size = 8)
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  o <- fromJSON(file.path(dir, "options.json"))
  samples <- read.delim(file.path(dir, "samples.tsv"), stringsAsFactors = TRUE)
  X <- model.matrix(as.formula(o$formula), samples)
  data <- matrix(rd(dir, "data"), ncol = nrow(X))
  w <- rd(dir, "weights")
  if (length(w) > max(dim(data))) w <- matrix(w, ncol = nrow(X))
  lib <- rd(dir, "lib")
  vd <- if (!is.null(o$var_design)) as.matrix(o$var_design)
  span <- if (is.null(o$span)) 0.5 else o$span
  caught <- character(0)
  keep <- function(c) { caught <<- c(caught, conditionMessage(c))
    invokeRestart(if (inherits(c, "warning")) "muffleWarning" else "muffleMessage") }
  failed <- function(e) { writeLines(conditionMessage(e), file.path(dir, "out_error.txt")); NULL }
  run <- function(expr) tryCatch(withCallingHandlers(expr, warning = keep, message = keep), error = failed)
  fit <- NULL
  # lmFit prints the coefficients it cannot estimate; the output is not needed.
  printed <- capture.output({
    if (o$kind == "weights") {
      aw <- run(arrayWeights(data, X, weights = w, var.design = vd, var.group = o$var_group,
                             prior.n = o$prior_n, method = o$method))
      if (!is.null(aw)) wr(dir, "sample.weights", aw)
    } else if (o$kind == "voom") {
      v <- run(voom(data, X, lib.size = lib, weights = w, span = span,
                    adaptive.span = is.null(o$span), save.plot = TRUE))
      # as.vector drops the attribute that marks asMatrixWeights' result as array
      # weights, which lmFit would otherwise take from the first gene.
      W <- v$weights * as.vector(asMatrixWeights(w, dim(data)))
      wr(dir, "E", v$E); wr(dir, "weights", W)
      wr(dir, "xy.x", v$voom.xy$x); wr(dir, "xy.y", v$voom.xy$y)
      wr(dir, "line.x", v$voom.line$x); wr(dir, "line.y", v$voom.line$y)
      if (!is.null(v$span)) wr(dir, "span", v$span)
      if (is.null(o$deviation)) fit <- lmFit(v$E, X, weights = W)
    } else {
      y <- if (isTRUE(o$tmm)) calcNormFactors(DGEList(data, lib.size = lib)) else data
      if (isTRUE(o$tmm)) lib <- NULL
      v <- run(voomWithQualityWeights(y, X, lib.size = lib, var.design = vd, var.group = o$var_group,
                                      method = o$method, span = span, adaptive.span = is.null(o$span)))
      wr(dir, "E", v$E); wr(dir, "weights", v$weights); wr(dir, "sample.weights", v$targets$sample.weights)
      if (!is.null(v$span)) wr(dir, "span", v$span)
      if (nrow(X) > ncol(X)) fit <- lmFit(v, X)
    }
    if (!is.null(fit)) {
      # makeContrasts renames (Intercept) in its row names, with a warning.
      if (length(o$contrasts)) fit <- contrasts.fit(fit, suppressWarnings(
        makeContrasts(contrasts = unlist(o$contrasts), levels = X)))
      # The default prior estimator, as CytePool's: zero weights make the residual
      # df unequal, where it is the likelihood estimator (fitFDistUnequalDF1).
      eb <- suppressWarnings(eBayes(fit))
      wr(dir, "t", eb$t); wr(dir, "p.value", eb$p.value)
    }
  })
  writeLines(caught, file.path(dir, "out_warnings.txt"))
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR"))), file.path(root, "versions.txt"))
"""


def simulate_counts(rng, n_genes, groups, sizes, noisy=()):
    """Samples x genes negative binomial counts with group effects; the samples in
    ``noisy`` get extra log-normal noise (sd 0.6), as pseudobulk samples pooled from
    few cells or from an outlying donor do."""
    levels, index = np.unique(groups, return_inverse=True)
    proportion = np.exp(rng.normal(0.0, 2.0, n_genes))
    proportion /= proportion.sum()
    lfc = np.zeros((levels.size, n_genes))
    changed = rng.random(n_genes) < 0.1
    lfc[1:, changed] = rng.normal(0.0, 1.5, (levels.size - 1, changed.sum()))
    mean = sizes[:, np.newaxis] * proportion * 2.0 ** lfc[index]
    for sample in noisy:
        mean[sample] *= np.exp(rng.normal(0.0, 0.6, n_genes))
    size = 1.0 / (0.05 * np.exp(rng.normal(0.0, 0.5, n_genes)))
    return rng.negative_binomial(size, size / (size + mean)).astype(float)


def simulate_values(rng, n_genes, scales):
    """Samples x genes normal log-expression values whose gene means and standard
    deviations vary; sample i's noise is multiplied by scales[i]."""
    means = rng.normal(6.0, 2.0, n_genes)
    sds = np.exp(rng.normal(-1.0, 0.4, n_genes))
    noise = rng.normal(0.0, 1.0, (len(scales), n_genes))
    return means + np.asarray(scales, dtype=float)[:, np.newaxis] * sds * noise


def make_cases(rng, small):
    """Cases by name; ``small`` gives the sizes of the test fixtures."""
    genes, many = (100, 200) if small else (2_000, 20_000)
    two = ({"group": np.repeat(["A", "B"], 4)}, "~ group")
    ages = rng.uniform(20, 70, 12).round(1)
    twelve = ({"group": np.repeat(list("ABC"), 4), "age": ages}, "~ group + age")
    one = ({"group": np.repeat(["A"], 4)}, "~ 1")
    saturated = ({"sample": np.array([f"S{i}" for i in range(4)])}, "~ sample")
    noisy = simulate_values(rng, genes, [1, 1, 1, 1, 1, 1, 3, 1])
    mixed = simulate_values(rng, genes, [1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1])
    values = simulate_values(rng, genes, [1, 1, 1, 1, 1, 1, 3, 1])
    observation = rng.uniform(0.2, 3.0, values.shape)
    missing = np.where(rng.random(values.shape) < 0.05, np.nan, values)
    missing[:6, 5] = np.nan  # two samples of one group left: one residual df
    missing[[2, 3, 5, 6, 7], 6] = np.nan  # two of one group, one of the other
    missing[:, 7] = 3.0  # no residual variance
    missing[:, 8] = np.repeat([1.0, 2.0], 4)  # fitted exactly
    counts = simulate_counts(rng, genes, two[0]["group"], rng.uniform(1e6, 3e6, 8), [6])
    flat = voom(counts, Design(*two), span=0.5)
    sizes = rng.uniform(1e6, 3e6, 8)
    counts = simulate_counts(rng, genes, two[0]["group"], sizes)
    weights = rng.uniform(0.3, 2.0, counts.shape)
    zeros = weights * (rng.random(counts.shape) > 0.05)
    twelve_counts = simulate_counts(
        rng, genes, twelve[0]["group"], rng.uniform(1e6, 3e6, 12)
    )
    sample_weights = rng.uniform(0.3, 2.0, 12)
    poor = simulate_counts(rng, genes, two[0]["group"], sizes, [2, 6])
    poor_twelve = simulate_counts(
        rng, genes, twelve[0]["group"], rng.uniform(5e5, 5e6, 12), [3, 9]
    )
    lib = poor_twelve.sum(axis=1) * rng.uniform(0.8, 1.2, 12)
    contrasts = {"BvsA": "groupB", "CvsB": "groupC - groupB"}
    covariate = np.linspace(0.0, 1.0, 12)[:, np.newaxis] ** 2
    # Data whose REML criterion rises after three iterations, where limma stops.
    early = np.random.default_rng(13)
    rises = (
        {"group": twelve[0]["group"], "x": early.uniform(0, 10, 12)},
        "~ group + x",
    )
    spread = np.exp(early.normal(0.0, 0.5, 12))[:, np.newaxis]
    oscillating = early.normal(5.0, 2.0, 200) + spread * np.exp(
        early.normal(-1.0, 0.5, 200)
    ) * early.normal(0.0, 1.0, (12, 200))
    both = ("gene_by_gene", "reml")
    table = [  # name, kind, design, data, settings; a tuple of methods gives a case each
        ("one_noisy_sample", "weights", two, noisy, {"method": both}),
        ("groups_covariate", "weights", twelve, mixed, {"method": both}),
        ("variance_groups", "weights", twelve, mixed, {"method": both, "var_group": ["a", "a", "b", "b"] * 3}),
        ("variance_covariate", "weights", twelve, mixed, {"method": both, "var_design": covariate}),
        ("prior_n_0", "weights", two, noisy, {"method": both, "prior_n": 0.0}),
        ("prior_n_50", "weights", two, noisy, {"method": both, "prior_n": 50.0}),
        ("observation_weights", "weights", two, values, {"method": ("auto", "reml"), "weights": observation}),
        ("missing_values", "weights", two, missing, {"method": ("auto", "reml")}),
        ("intercept_only", "weights", one, simulate_values(rng, genes, [1, 1, 1, 2]), {}),
        ("reml_criterion_rises", "weights", rises, oscillating, {"method": "reml"}),
        ("one_gene", "weights", two, values[:, :1], {"method": "gene_by_gene"}),
        ("no_replication", "weights", saturated, values[:4], {"method": "reml"}),
        ("voom_log_cpm", "weights", two, flat.log2_cpm, {"method": both, "weights": flat.weights}),
        (f"{many}_genes", "weights", two, simulate_values(rng, many, [1] * 6 + [3, 1]), {"method": both}),
        ("voom_sample_weights", "voom", two, counts, {"weights": sample_weights[:8]}),
        ("voom_observation_weights", "voom", two, counts, {"weights": weights}),
        ("voom_feature_weights", "voom", two, counts, {"weights": weights[:1]}),
        ("voom_zero_weights", "voom", two, counts, {"weights": zeros}),
        ("voom_adaptive_span", "voom", twelve, twelve_counts, {"weights": sample_weights, "span": None}),
        ("quality", "quality", two, poor, {"method": both}),
        ("quality_adaptive_span", "quality", two, poor, {"span": None}),
        ("quality_variance_groups", "quality", two, poor, {"var_group": [1, 1, 2, 2] * 2}),
        ("quality_tmm_contrasts", "quality", twelve, poor_twelve, {"method": "reml", "tmm": True, "lib": lib, "contrasts": contrasts}),
        ("quality_variance_covariate", "quality", twelve, poor_twelve, {"var_design": np.log(lib)[:, np.newaxis], "span": None}),
        (f"quality_{many}_genes", "quality", two, simulate_counts(rng, many, two[0]["group"], sizes, [2, 6]), {"span": None}),
        ("quality_no_replication", "quality", saturated, poor[:4], {}),
    ]  # fmt: skip
    cases = {}
    for name, kind, (samples, formula), data, settings in table:
        # limma's defaults: "auto" for arrayWeights, "genebygene" for quality weights.
        method = settings.get("method", "gene_by_gene" if kind == "quality" else "auto")
        for m in method if isinstance(method, tuple) else (method,):
            full = f"{name}_{m}" if isinstance(method, tuple) else name
            cases[full] = {"prior_n": 10.0, "span": 0.5, "tmm": False} | settings
            cases[full] |= {"kind": kind, "samples": samples, "formula": formula}
            cases[full] |= {"data": data, "method": m}
    return cases


def deviation_cases(rng):
    """Inputs where CytePool deliberately differs from limma, with what each does."""
    two = {"group": np.repeat(["A", "B"], 4)}
    values = simulate_values(rng, 200, [1, 1, 1, 1, 1, 1, 3, 1])
    single = values.copy()
    single[0, 1:] = np.nan  # every gene but the first has a missing value
    counts = simulate_counts(rng, 300, two["group"], rng.uniform(1e6, 3e6, 8))
    unused = rng.uniform(0.3, 2.0, counts.shape)
    unused[:, 0] = 0.0
    inestimable = rng.uniform(0.3, 2.0, counts.shape)
    inestimable[:4, 1] = 0.0
    group = {"var_group": [1, 1, 1, 1, 2, 2, 2, 2]}
    alone = {"group": np.array(["A"] * 7 + ["B"])}  # the last sample alone in B
    # kind, data, settings; what limma and CytePool do, whether limma fails, and the
    # error CytePool raises (None: it returns values).
    cases = {
        "variance design and groups together": ("weights", values, group | {"var_design": np.eye(8)[:, :1]}, "uses var.group", "refuses the two", False, "not both"),
        "negative prior.n": ("weights", values, {"prior_n": -1.0}, "accepts it", "refuses it", False, "prior_n must be"),
        "REML with one complete gene": ("weights", single, {"method": "reml"}, "fails", "weights of one and a warning", True, None),
        "REML with prior.n 0 and a sample without information": ("weights", values, {"method": "reml", "prior_n": 0.0, "samples": alone}, "weights that rounding error decides", "refuses them", False, "information about the sample weights is"),
        "voom: a gene whose prior weights are all zero": ("voom", counts, {"weights": unused}, "a trend through its missing standard deviation", "the gene left out of the trend", False, None),
        "voom: zero weights leave a gene's coefficients inestimable": ("voom", counts, {"weights": inestimable}, "missing weights for the gene", "positive weights where the prior is", False, None),
    }  # fmt: skip
    base = {"method": "auto", "prior_n": 10.0, "span": 0.5, "tmm": False}
    return {
        name: base | {"kind": kind, "samples": two, "formula": "~ group"} | settings
        | {"data": data, "deviation": deviation}
        for name, (kind, data, settings, *deviation) in cases.items()
    }  # fmt: skip


def run_limma(cases):
    """limma's outputs and messages for each case; returns the versions of R, limma
    and edgeR."""
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
            for key in ("data", "weights", "lib"):
                if key in case:
                    path = directory / f"{key}.bin"
                    path.write_bytes(np.ascontiguousarray(case[key]).tobytes())
            keys = ("kind", "formula", "prior_n", "span", "tmm", "var_group")
            options = {k: case.get(k) for k in (*keys, "contrasts", "deviation")}
            options["method"] = R_METHODS[case["method"]]
            if case.get("var_design") is not None:
                options["var_design"] = np.asarray(case["var_design"]).tolist()
            (directory / "options.json").write_text(json.dumps(options))
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        command = [rscript, "--vanilla", str(root / "compare.R"), str(root)]
        subprocess.run(command, check=True)
        for name, case in cases.items():
            directory = root / names[name]
            case["limma"] = {
                path.stem[4:]: np.fromfile(path, dtype="<f8")
                for path in directory.glob("out_*.bin")
            }
            text = (directory / "out_warnings.txt").read_text().strip()
            case["limma warnings"] = text.splitlines()
            case["limma failed"] = (directory / "out_error.txt").exists()
        return (root / "versions.txt").read_text().split("\n")[:3]


def run_cytepool(case):
    """CytePool's result for a case, its design, and the warnings it gave."""
    design = Design(case["samples"], case["formula"], contrasts=case.get("contrasts"))
    variance = {"variance_design": case.get("var_design")}
    variance["variance_groups"] = case.get("var_group")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if case["kind"] == "weights":
            result = sample_quality_weights(
                case["data"], design, weights=case.get("weights"), method=case["method"],
                prior_n=case["prior_n"], **variance,
            )  # fmt: skip
        elif case["kind"] == "voom":
            result = voom(
                case["data"], design, library_sizes=case.get("lib"),
                weights=case.get("weights"), span=case["span"],
            )  # fmt: skip
        else:
            lib = case.get("lib")
            factors = (
                tmm_factors(case["data"], library_sizes=lib) if case["tmm"] else None
            )
            result = voom_with_quality_weights(
                case["data"], design, library_sizes=lib, norm_factors=factors,
                span=case["span"], method=case["method"], **variance,
            )  # fmt: skip
    return design, result, [str(w.message) for w in caught]


def cytepool_results(case):
    """CytePool's outputs for a case, keyed and shaped as limma writes them, and the
    warnings it gave."""
    design, result, caught = run_cytepool(case)
    if case["kind"] == "weights":
        return {"sample.weights": result}, caught
    out = {"E": result.log2_cpm, "weights": result.weights}
    if result.sample_weights is not None:
        out["sample.weights"] = result.sample_weights
    if case["span"] is None and not np.isnan(result.span):
        out["span"] = result.span  # limma returns no span without replication
    in_trend = ~np.isnan(result.sqrt_sd)
    if case["kind"] == "voom":
        x = result.mean_log_count[in_trend]
        order = np.argsort(x, kind="stable")
        out |= {"xy.x": x, "xy.y": result.sqrt_sd[in_trend], "line.x": x[order]}
        out["line.y"] = result.sqrt_sd_trend[in_trend][order]
    if in_trend.any():
        fit = fit_linear_model(result.log2_cpm, design, weights=result.weights)
        if len(design.contrasts):
            fit = contrasts_fit(fit, design.contrasts, method="limma")
        moderated = moderated_statistics(fit)
        # R writes matrices column by column.
        out |= {"t": moderated.t.T, "p.value": moderated.p_value.T}
    return out, caught


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


def compare(cases):
    """Rows of (check, reference, metric, value, tolerance, result)."""
    table = []
    for name, case in cases.items():
        if "deviation" in case:
            table += deviation_rows(name, case)
            continue
        ours, caught = cytepool_results(case)
        limma, kind = case["limma"], case["kind"]
        source = REFERENCES.get(kind, "voomWithQualityWeights")
        for group, keys in GROUPS.items():
            exact = group in EXACT_GROUPS or kind == "voom"
            tolerance = EXACT if exact else CLOSE
            values = [
                difference(ours.get(k), limma.get(k, [np.nan]), floor, k == "p.value")
                for k, floor in keys.items()
                if k in limma or k in ours
            ]
            if values:
                result = "pass" if max(values) <= tolerance else "FAIL"
                reference = source + (", lmFit, eBayes" if "t, p" in group else "")
                cells = (f"{name}: {group}", reference, "max scaled difference")
                table.append((*cells, f"{max(values):.1e}", f"{tolerance:.0e}", result))
        # limma reports the genes REML leaves out, and designs without replication.
        for message in dict.fromkeys(case["limma warnings"]):
            removed = re.search(r"removing (\d+) rows", message)
            reports = {"no replication": "no replication", "prematurely": "rose"}
            said = next((v for k, v in reports.items() if k in message), None)
            if removed or said:
                said = f"leaves out {removed[1]} of" if removed else said
                agree = any(said in w for w in caught)
                outcome = ("both", "pass") if agree else ("limma only", "FAIL")
                cells = (f"{name}: report", source, "who reports it", outcome[0])
                table.append((*cells, "n/a", outcome[1]))
    return table


def deviation_rows(name, case):
    limma_does, ours_does, limma_fails, message = case["deviation"]
    try:
        _, result, _ = run_cytepool(case)
        refused = None
    except ValueError as error:
        result, refused = None, str(error)
    expected = case["limma failed"] == limma_fails and (
        refused is None if message is None else message in (refused or "")
    )
    rows = []
    if "all zero" in name and result is not None:
        # CytePool leaves the gene out of the trend; the other genes get the weights
        # limma gives them without it (with the library sizes of every gene).
        reduced = {k: v for k, v in case.items() if k not in ("deviation", "limma")}
        reduced |= {"data": case["data"][:, 1:], "weights": case["weights"][:, 1:]}
        reduced["lib"] = case["data"].sum(axis=1)
        run_limma({"reduced": reduced})
        value = difference(result.weights[:, 1:], reduced["limma"]["weights"])
        outcome = "pass" if value <= EXACT else "FAIL"
        cells = (f"{name}: the other genes", "voom without the gene", "max difference")
        rows.append((*cells, f"{value:.1e}", f"{EXACT:.0e}", outcome))
    if "one complete gene" in name and result is not None:
        expected &= bool(np.allclose(result, 1.0, rtol=0.0, atol=1e-15))
    if "inestimable" in name and result is not None:
        weights, prior = result.weights[:, 1], case["weights"][:, 1]
        expected &= bool(np.all((weights > 0) == (prior > 0)))
        expected &= bool(np.isnan(case["limma"]["weights"]).any())
    value = f"limma: {limma_does}; CytePool: {ours_does}"
    outcome = "deviation" if expected else "FAIL"
    return [(f"deviation: {name}", "limma", "behaviour", value, "n/a", outcome), *rows]


def write_fixtures(cases):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    nested = ("samples", "contrasts", "limma", "limma failed")
    for name, case in cases.items():
        arrays = {k: np.asarray(v) for k, v in case.items() if k not in nested}
        arrays = {k: v for k, v in arrays.items() if v.dtype != object}  # None
        arrays |= {f"samples {k}": np.asarray(v) for k, v in case["samples"].items()}
        if case.get("contrasts"):
            arrays["contrast names"] = np.array(list(case["contrasts"]))
            arrays["contrast expressions"] = np.array(list(case["contrasts"].values()))
        arrays |= {f"limma {k}": v for k, v in case["limma"].items()}
        arrays["limma warnings"] = np.array(case["limma warnings"], dtype=str)
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(1), small=False)
    cases |= deviation_cases(np.random.default_rng(3))
    # The fixtures' cases, compared too; run_limma adds limma's outputs to them.
    small = make_cases(np.random.default_rng(2), small=True)
    if arguments.write_fixtures:
        cases |= {f"{name}_fixture": case for name, case in small.items()}
    versions = run_limma(cases)
    print("Reference: " + ", ".join(versions))
    table = compare(cases)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        write_fixtures(small)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
