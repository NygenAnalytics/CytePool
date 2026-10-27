"""Check what the figures of cytepool.plotting show against what R draws.

Each figure that follows an R function is drawn by CytePool and by R on the same data,
and the values that R's graphics received are compared with those of the drawn
artists. R's base graphics are read from the recorded display list of the plot
(``recordPlot()``), which holds every primitive drawn with its coordinates, and
speckle's ggplot from its built layer data; R runs as a black box.

Data (seed 20261101): counts of 300 genes in 8 samples, 4 controls and 4 stimulated,
negative binomial with about 15% of the genes changed, 2 genes without counts and 4
with an outlying count.

- ``plot_mds`` of their log2 counts per million against ``plotMDS(pch = 16)``: the
  leading log2 fold changes over the top 50 genes, over 500 (at least the number of
  genes: a principal component plot), and over the 100 genes of largest variance
  (``gene.selection = "common"``, dimensions 2 and 3); and of a cube of the counts in
  which two samples have fewer than 10 cells, against edgeR's ``plotMDS`` of the
  others after ``filterByExpr`` and ``normLibSizes``. Coordinates are compared up to
  the sign of each dimension, and the percentages of the axis labels must be equal.
- ``plot_mean_variance`` against ``voom(save.plot = TRUE)``, and ``plot_sample_weights``
  against the bars of ``voomWithQualityWeights(plot = TRUE)``, with ``adaptive.span =
  TRUE`` as CytePool chooses the span.
- ``plot_residual_variances`` against ``plotSA``: the log2 counts per million with six
  hypervariable genes, moderated with a trend and robustly (the outliers it marks are
  compared too), and without either.
- ``plot_md``, ``plot_volcano`` and ``plot_p_values`` of the table of ``limma_voom``
  against ``plotMD(status = decideTests(fit))``, ``volcanoplot(highlight = 10)`` and
  ``hist(breaks = seq(0, 1, 0.05))`` of limma-voom run as the limma User's Guide runs
  it.
- ``plot_barcode`` against ``barcodeplot`` on the same moderated t-statistics: a set of
  25 genes with the default span of the worm, and 7 of the first 60 genes with span 1.
- ``plot_dispersions`` against ``plotDispEsts`` after ``DESeq``, and
  ``plot_quasi_dispersions`` against ``plotQLDisp`` after ``glmQLFit`` with the same
  TMM factors. CytePool's estimates differ from DESeq2's and edgeR's by the deliberate
  differences that deseq2_tests.py and quasi_likelihood.py report, so each figure is
  drawn from R's own estimates, as R drew them, and from CytePool's fit of the counts,
  for information.
- ``plot_forest`` against metafor's ``forest``: three studies of 4, 5 and 6 donors,
  tested by ``limma_voom`` (the third does not measure 5 genes) and meta-analysed by
  ``meta_analysis`` by REML with z-tests and by DerSimonian-Laird with the
  Knapp-Hartung adjustment. For three genes, one tested by two studies, R refits
  ``rma.uni`` on the studies' log2 fold changes and standard errors (REML to
  convergence, ``control = list(threshold = 1e-13)``); the studies' intervals are
  compared with those forest draws, and the pooled estimate and interval with those it
  writes beside its diamond, which it clips at the axis limits.
- ``plot_composition`` against speckle's ``plotCellTypeProps``, and
  ``plot_proportions`` against the proportions, ``PropMean`` and ``FDR`` of
  ``propeller``, on the cell numbers of 5 cell types in 12 samples of two conditions.

Differences are scaled by max(|reference|, 1). Tiers: exact (1e-10) for values both
programs compute by the same deterministic steps; close (1e-6) for iterative fits
(robust moderation, sample weights, REML). Counts per bin, the genes highlighted,
outliers and labels must be identical. R writes numbers with 15 significant digits.

Requirements:
    Python 3.13 with numpy, numba and matplotlib, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with limma, edgeR, DESeq2 and jsonlite (``RSCRIPT``), metafor and
    jsonlite (``METAFOR_RSCRIPT``), and speckle, ggplot2 and jsonlite
    (``SPECKLE_RSCRIPT``); each variable names an Rscript executable and defaults to
    ``Rscript``. R runs with 8 GB of address space at most.

Usage:
    PYTHONPATH=. python validation/plotting.py
    PYTHONPATH=. python validation/plotting.py --write-fixtures
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from cytepool import (
    Cube,
    Design,
    from_bulk,
    limma_voom,
    meta_analysis,
    plotting,
    propeller,
)
from cytepool.normalization import log2_cpm, tmm_factors
from cytepool.stats.deseq2 import fit_deseq2
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.quasi_likelihood import fit_quasi_likelihood
from cytepool.stats.voom import voom, voom_with_quality_weights

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "plotting"
EXACT = 1e-10
CLOSE = 1e-6
MEMORY = 8_000_000_000  # bytes of address space for R
MDS_CASES = {"mds1": (50, "pairwise", (1, 2)), "mds2": (500, "pairwise", (1, 2))}
MDS_CASES["mds3"] = (100, "common", (2, 3))
FOREST_CASES = (("reml", "z"), ("dl", "knapp_hartung"))
STUDIES = (("A", 4, 300), ("B", 5, 300), ("C", 6, 295))  # name, donors, genes

DRAWN = r"""
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(name) readBin(file.path(root, name), "double",
                             n = file.size(file.path(root, name)) / 8, size = 8)
# The primitives a base-graphics plot drew, with their numeric and text arguments.
drawn <- function(expr) {
  pdf(NULL)
  dev.control("enable")
  force(expr)
  dl <- recordPlot()[[1]]
  invisible(dev.off())
  lapply(dl, function(e) {
    f <- e[[2]][[1]]
    list(name = if (is.list(f) && !is.null(f$name)) f$name else "",
         args = unname(lapply(e[[2]][-1], function(a) {
           if (is.list(a)) a <- a[vapply(a, function(v) is.numeric(v) || is.character(v), TRUE)]
           if (is.numeric(a) || is.character(a) || is.list(a)) a else NULL
         })))
  })
}
out <- list()
save <- function(versions) {
  out$versions <- versions
  jsonlite::write_json(out, file.path(root, "out.json"), digits = NA, auto_unbox = TRUE,
                       null = "null", na = "null")
}
"""

R_LIMMA = (
    DRAWN
    + r"""
suppressPackageStartupMessages({library(limma); library(edgeR); library(DESeq2)})
counts <- matrix(rd("counts.bin"), ncol = 8)  # genes x samples
rownames(counts) <- readLines(file.path(root, "genes.txt"))
condition <- factor(rep(c("ctrl", "stim"), each = 4))
X <- model.matrix(~ condition)
E <- matrix(rd("E.bin"), ncol = 8)
cases <- list(mds1 = list(50, "pairwise", 1:2), mds2 = list(500, "pairwise", 1:2),
              mds3 = list(100, "common", 2:3))
for (k in names(cases)) out[[k]] <- drawn(plotMDS(E, top = cases[[k]][[1]],
  gene.selection = cases[[k]][[2]], dim.plot = cases[[k]][[3]], pch = 16))
y <- DGEList(counts[, as.logical(rd("used.bin"))])
y <- normLibSizes(y[suppressWarnings(filterByExpr(y)), , keep.lib.sizes = FALSE])
out$mds_cube <- drawn(plotMDS(y, pch = 16))
v <- voom(counts, X, save.plot = TRUE, adaptive.span = TRUE)
out$voom <- list(xy = v$voom.xy[c("x", "y")], line = v$voom.line)
out$weights <- drawn(voomWithQualityWeights(counts, X, plot = TRUE, adaptive.span = TRUE))
noisy <- matrix(rd("noisy.bin"), ncol = 8)
out$sa_robust <- drawn(plotSA(eBayes(lmFit(noisy, X), trend = TRUE, robust = TRUE)))
out$sa_plain <- drawn(plotSA(eBayes(lmFit(E, X))))
dge <- DGEList(counts)
dge <- calcNormFactors(dge[filterByExpr(dge, design = X), , keep.lib.sizes = FALSE])
fit <- eBayes(lmFit(voom(dge, X, adaptive.span = TRUE), X))
out$md <- drawn(plotMD(fit, column = 2, status = decideTests(fit)[, 2]))
out$volcano <- drawn(volcanoplot(fit, coef = 2, highlight = 10, names = rownames(dge)))
out$histogram <- hist(fit$p.value[, 2], breaks = seq(0, 1, 0.05), plot = FALSE)$counts
t <- rd("t.bin")
out$barcode <- drawn(barcodeplot(t, index = as.integer(rd("set.bin"))))
out$barcode_span <- drawn(barcodeplot(t[1:60], index = as.integer(rd("set_span.bin")),
                                      span.worm = 1))
dds <- DESeqDataSetFromMatrix(counts, data.frame(condition = condition), ~ condition)
dds <- suppressMessages(DESeq(dds, quiet = TRUE))
out$dispersions <- drawn(plotDispEsts(dds))
m <- mcols(dds)
out$deseq2 <- list(base_mean = m$baseMean, genewise = m$dispGeneEst, trended = m$dispFit,
                   dispersion = m$dispersion, outlier = m$dispOutlier, all_zero = m$allZero)
ql <- glmQLFit(DGEList(counts, norm.factors = rd("norm_factors.bin")), X)
out$ql <- drawn(plotQLDisp(ql))
out$ql_values <- list(covariate = ql$AveLogCPM, deviance = ql$deviance.adj,
                      df = ql$df.residual.adj, posterior = ql$s2.post, prior = ql$s2.prior)
save(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
       paste("edgeR", packageVersion("edgeR")), paste("DESeq2", packageVersion("DESeq2"))))
"""
)

R_METAFOR = (
    DRAWN
    + r"""
suppressPackageStartupMessages(library(metafor))
cases <- jsonlite::read_json(file.path(root, "forest.json"), simplifyVector = TRUE)
for (case in names(cases)) {
  s <- cases[[case]]
  fit <- rma.uni(s$yi, sei = s$sei, method = c(reml = "REML", dl = "DL")[[s$method]],
                 test = c(z = "z", knapp_hartung = "knha")[[s$test]],
                 control = list(threshold = 1e-13, maxiter = 10000))
  # The diamond stops at the axis limits, so the pooled estimate and interval that
  # forest writes beside it are taken from the fit.
  out[[case]] <- list(drawn = drawn(forest(fit)),
                      pooled = c(fit$ci.lb, fit$beta[1], fit$ci.ub))
}
save(c(paste("R", getRversion()), paste("metafor", packageVersion("metafor"))))
"""
)

R_SPECKLE = r"""
suppressPackageStartupMessages({library(speckle); library(limma); library(ggplot2)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
cells <- jsonlite::read_json(file.path(root, "cells.json"), simplifyVector = TRUE)
built <- ggplot_build(plotCellTypeProps(clusters = cells$cluster, sample = cells$sample))
fill <- built$plot$scales$get_scales("fill")
layer <- built$data[[1]]
props <- getTransformedProps(cells$cluster, cells$sample, transform = "logit")
result <- suppressMessages(propeller(clusters = cells$cluster, sample = cells$sample,
                                     group = cells$group))
out <- list(
  bar_sample = built$layout$panel_params[[1]]$x$get_labels()[layer$x],
  bar_cluster = fill$get_limits()[match(layer$fill, fill$map(fill$get_limits()))],
  bar_height = layer$ymax - layer$ymin,
  clusters = rownames(props$Proportions), samples = colnames(props$Proportions),
  proportions = unname(unclass(props$Proportions)),
  table_clusters = rownames(result), mean_ctrl = result$PropMean.ctrl,
  mean_stim = result$PropMean.stim, fdr = result$FDR,
  versions = c(paste("R", getRversion()), paste("speckle", packageVersion("speckle"))))
jsonlite::write_json(out, file.path(root, "out.json"), digits = NA, auto_unbox = TRUE,
                     matrix = "rowmajor")
"""


def run_r(program, inputs, variable):
    """Run an R program on input files and return its JSON output. A samples x genes
    array written as binary is read by R as genes x samples."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, value in inputs.items():
            if name.endswith(".bin"):
                data = np.ascontiguousarray(value, dtype="<f8").tobytes()
                (root / name).write_bytes(data)
            elif name.endswith(".json"):
                (root / name).write_text(json.dumps(value))
            else:
                (root / name).write_text("\n".join(value) + "\n")
        (root / "program.R").write_text(program)
        limit = (MEMORY, MEMORY)
        subprocess.run(
            [os.environ.get(variable, "Rscript"), "--vanilla", "program.R", "."],
            check=True,
            cwd=root,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, limit),
        )
        return json.loads((root / "out.json").read_text())


# What R drew.


def primitives(drawn, name, **where):
    """The arguments of the primitives of one name whose arguments match ``where``,
    given as a<position from 1>=value."""
    return [
        entry["args"]
        for entry in drawn
        if entry["name"] == name
        and all(entry["args"][int(k[1:]) - 1] == v for k, v in where.items())
    ]


def xy(args):
    """The x and y of a plotXY primitive; R writes a single value as a number."""
    return tuple(np.atleast_1d(np.array(args[0][k], dtype=float)) for k in "xy")


def series(drawn, color, kind="p"):
    """The x, y and arguments of the first plotXY of one colour, not a legend's."""
    for args in primitives(drawn, "C_plotXY", a2=kind):
        if args[4] == color and np.size(args[0]["x"]) > 3:
            return (*xy(args), args)
    raise ValueError(f"R drew no series in {color}")


def pairs(x, y):
    """Points as rows (x, y), sorted by x and then y."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    order = np.lexsort((y, x))
    return np.column_stack([x[order], y[order]])


def percentages(drawn):
    return [text.rsplit("(", 1)[1] for text in primitives(drawn, "C_title")[0][2:4]]


def column(values):
    return np.array(values, dtype=float)  # R's NA is written as null, read as NaN


def references(limma, metafor, speckle):
    """R's values, by name: what the checks and the tests compare with."""
    ref = {}
    for key in (*MDS_CASES, "mds_cube"):
        args = primitives(limma[key], "C_plotXY", a2="p")[0]
        ref[f"{key} points"] = np.column_stack(xy(args))
        ref[f"{key} percentages"] = np.array(percentages(limma[key]))
    voom_result = limma["voom"]
    ref["voom points"] = np.column_stack(
        [voom_result["xy"]["x"], voom_result["xy"]["y"]]
    )
    ref["voom line"] = np.column_stack(
        [voom_result["line"]["x"], voom_result["line"]["y"]]
    )
    ref["weights"] = column(primitives(limma["weights"], "C_rect")[-1][3])
    for key in ("sa_robust", "sa_plain"):
        args = primitives(limma[key], "C_plotXY", a2="p")[0]
        x, y = xy(args)
        red = np.array(args[4] if isinstance(args[4], list) else [args[4]]) == "red"
        ref[f"{key} points"] = pairs(x, y)
        ref[f"{key} outliers"] = (
            pairs(x[red], y[red]) if red.size == x.size else pairs([], [])
        )
        lines = primitives(limma[key], "C_plotXY", a2="l")
        abline = [primitives(limma[key], "C_abline")[0][2]] * 2 if not lines else None
        ref[f"{key} prior"] = np.column_stack(xy(lines[0])) if lines else column(abline)
    for gid, color in (("up", "red"), ("down", "blue"), ("not significant", "black")):
        found = [
            a for a in primitives(limma["md"], "C_plotXY", a2="p") if a[4] == color
        ]
        ref[f"md {gid}"] = pairs(*xy(found[0])) if found else pairs([], [])
    ref["volcano points"] = pairs(
        *xy(primitives(limma["volcano"], "C_plotXY", a2="p")[0])
    )
    ref["volcano labels"] = np.sort(primitives(limma["volcano"], "C_text")[0][1])
    ref["histogram"] = np.array(limma["histogram"], dtype=np.int64)
    for key in ("barcode", "barcode_span"):
        bars = primitives(limma[key], "C_segments", a2=0, a4=0.5)[0][0]
        ref[f"{key} bars"] = np.sort(np.atleast_1d(column(bars)))
        ref[f"{key} worm"] = xy(primitives(limma[key], "C_plotXY", a2="l")[0])[1]
    for name, values in limma["deseq2"].items():
        ref[f"deseq2 {name}"] = column(values)
    drawn = limma["dispersions"]
    fx, fy, final = series(drawn, "dodgerblue")
    open_circle = np.array(final[2]) == 1
    ref["deseq2 drawn gene-wise"] = pairs(*series(drawn, "black")[:2])
    ref["deseq2 drawn final"] = pairs(fx, fy)
    ref["deseq2 drawn outliers"] = pairs(fx[open_circle], fy[open_circle])
    ref["deseq2 drawn fitted"] = pairs(*series(drawn, "red")[:2])
    for name, values in limma["ql_values"].items():
        ref[f"ql {name}"] = column(values)
    drawn = limma["ql"]
    rx, ry, _ = series(drawn, "black")
    zero = ref["ql df"] == 0.0
    ref["ql drawn raw"] = pairs(rx[~zero], ry[~zero])
    ref["ql drawn raw without df"] = ry[zero]
    ref["ql drawn squeezed"] = pairs(*series(drawn, "red")[:2])
    ref["ql drawn trend"] = np.column_stack(series(drawn, "blue", "l")[:2])
    for case, value in metafor.items():
        if case == "versions":
            continue
        segments = [
            a
            for a in primitives(value["drawn"], "C_segments")
            if a[1] == a[3] and a[1] > 0
        ]
        segments.sort(key=lambda a: -a[1])  # the first study on top
        ref[f"forest {case} intervals"] = np.array([[a[0], a[2]] for a in segments])
        ref[f"forest {case} pooled"] = column(value["pooled"])
    for name in ("bar_sample", "bar_cluster", "clusters", "samples", "table_clusters"):
        ref[f"speckle {name}"] = np.array(speckle[name])
    for name in ("bar_height", "proportions", "mean_ctrl", "mean_stim", "fdr"):
        ref[f"speckle {name}"] = column(speckle[name])
    return ref


# What CytePool drew.


def axes():
    return Figure().subplots()


def artists(ax, gid):
    return [a for a in ax.get_children() if a.get_gid() == gid]


def points(ax, *gids):
    """The points of scatter collections, as pairs (x, y) sorted by x and then y."""
    offsets = [c.get_offsets() for gid in gids for c in artists(ax, gid)]
    stacked = np.concatenate(offsets) if offsets else np.empty((0, 2))
    return pairs(stacked[:, 0], stacked[:, 1])


def scaled(ours, reference):
    """Largest |ours - reference| over max(|reference|, 1); inf for unequal shapes."""
    ours, reference = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if ours.shape != reference.shape:
        return np.inf
    if ours.size == 0:
        return 0.0
    return float(np.max(np.abs(ours - reference) / np.maximum(np.abs(reference), 1.0)))


def row(check, reference, value, tolerance):
    """A row of the table: a difference within a tolerance, information when the
    tolerance is None, or True for values that must be identical."""
    if isinstance(value, bool):
        return (
            check,
            reference,
            "identical",
            "yes" if value else "no",
            "0",
            "pass" if value else "FAIL",
        )
    if tolerance is None:
        return (
            check,
            reference,
            "max scaled difference",
            f"{value:.1e}",
            "n/a",
            "information",
        )
    return (
        check,
        reference,
        "max scaled difference",
        f"{value:.1e}",
        f"{tolerance:.0e}",
        "pass" if value <= tolerance else "FAIL",
    )


# Data.


def simulate(rng):
    """Counts of 300 genes in 4 controls and 4 stimulated samples, samples x genes;
    2 genes have no counts and 4 an outlying count in the first sample."""
    stim = np.repeat([0.0, 1.0], 4)
    base = np.exp(rng.uniform(1.0, 11.0, 300) * np.log(2.0))
    effect = np.where(rng.random(300) < 0.15, rng.normal(0.0, 1.5, 300), 0.0)
    depth = rng.uniform(0.6, 1.4, 8)
    mean = base * depth[:, np.newaxis] * np.exp(effect * stim[:, np.newaxis])
    counts = rng.negative_binomial(8, 8.0 / (8.0 + mean))
    counts[:, :2] = 0
    counts[0, 20:24] *= 40
    return counts


def study_counts(rng, counts):
    """The counts of three studies of the stimulation, the third without 5 genes."""
    built = {}
    for name, n_donors, n_genes in STUDIES:
        mean = counts[[0, 4] * n_donors].astype(float) + 1.0
        mean = mean * rng.uniform(0.7, 1.3, (2 * n_donors, 1))
        built[name] = rng.negative_binomial(10, 10.0 / (10.0 + mean))[:, :n_genes]
    return built


def study_results(counts):
    """limma_voom of each study, paired by donor; the tests build them the same way."""
    results = {}
    for name, values in counts.items():
        donors = [f"{name}{d}" for d in range(values.shape[0] // 2)]
        samples = {
            "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
            "donor": [d for d in donors for _ in range(2)],
            "condition": ["ctrl", "stim"] * len(donors),
        }
        genes = [f"g{j:03d}" for j in range(values.shape[1])]
        cube = from_bulk(values, samples=samples, features=genes)
        design = Design(
            cube.samples,
            "~ donor + condition",
            levels={"condition": ["ctrl", "stim"]},
            contrasts={"stim": "conditionstim"},
        )
        results[name] = limma_voom(cube, design)
    return results


def forest_features(table):
    """The largest log2 fold change of three studies, the median p-value of three
    studies, and the first gene of two studies."""
    n, ids = np.asarray(table["n_studies"]), np.asarray(table["feature_id"].tolist())
    three = np.flatnonzero(n == 3)
    largest = three[np.argmax(np.abs(np.asarray(table["log2_fold_change"])[three]))]
    median = three[np.argsort(np.asarray(table["p_value"])[three])[three.size // 2]]
    return [str(ids[largest]), str(ids[median]), str(ids[np.flatnonzero(n == 2)[0]])]


def forest_inputs(results, features):
    """The log2 fold changes and standard errors of the studies that tested a gene."""
    cases = {}
    for method, test in FOREST_CASES:
        for feature in features:
            yi, sei = [], []
            for result in results.values():
                table = result.table("bulk", "stim")
                hit = np.flatnonzero(table["feature_id"] == feature)
                if hit.size:
                    yi.append(float(table["log2_fold_change"][hit[0]]))
                    sei.append(float(table["standard_error"][hit[0]]))
            cases[f"{method} {feature}"] = {
                "yi": yi,
                "sei": sei,
                "method": method,
                "test": test,
            }
    return cases


def composition(rng):
    """Cell numbers of 5 cell types in 6 control and 6 stimulated samples; the
    monocytes expand on stimulation."""
    share = np.array([0.45, 0.2, 0.1, 0.2, 0.05])
    cells = np.empty((12, 5), dtype=np.int64)
    for s in range(12):
        p = share * (np.array([1.0, 1.0, 1.0, 1.8, 1.0]) if s >= 6 else 1.0)
        cells[s] = rng.multinomial(
            int(rng.integers(600, 1200)), rng.dirichlet(60 * p / p.sum())
        )
    return cells


def composition_cube(cells):
    sample_ids = [f"S{i:02d}" for i in range(1, 13)]
    return Cube(
        {"counts": (cells.T[:, :, np.newaxis] > 0).astype(np.uint32)},
        samples={"sample_id": sample_ids, "condition": ["ctrl"] * 6 + ["stim"] * 6},
        groups={"group_id": ["T", "B", "NK", "Mono", "DC"]},
        features={"feature_id": ["x"]},
        n_cells=cells,
    )


# The checks.


def check_samples(ref, E, cube):
    rows = []
    cases = {
        key: ({"top": t, "gene_selection": s, "dimensions": d}, E)
        for key, (t, s, d) in MDS_CASES.items()
    }
    cases["mds_cube"] = ({"group": "T"}, cube)
    for key, (options, data) in cases.items():
        ax = plotting.plot_mds(data, ax=axes(), **options)
        ours = artists(ax, "samples")[0].get_offsets()
        theirs = ref[f"{key} points"]
        signs = np.sign(np.sum(ours * theirs, axis=0))
        value = scaled(ours * np.where(signs == 0.0, 1.0, signs), theirs)
        labels = [ax.get_xlabel().rsplit("(", 1)[1], ax.get_ylabel().rsplit("(", 1)[1]]
        if labels != list(ref[f"{key} percentages"]):
            value = np.inf
        label = (
            f"MDS, top {options['top']}, {options['gene_selection']}, dimensions "
            f"{options['dimensions']}"
            if data is E
            else "MDS of a cube, two samples with fewer than 10 cells, filterByExpr, TMM"
        )
        rows.append(
            row(
                f"{label}: coordinates up to sign, percentages", "plotMDS", value, EXACT
            )
        )
    return rows


def check_trends(ref, counts, E, noisy, design):
    ax = plotting.plot_mean_variance(voom(counts, design), ax=axes())
    value = max(
        scaled(artists(ax, "points")[0].get_offsets(), ref["voom points"]),
        scaled(artists(ax, "trend")[0].get_xydata(), ref["voom line"]),
    )
    rows = [
        row(
            "voom's mean-variance trend: points and LOWESS line",
            "voom(save.plot = TRUE)",
            value,
            EXACT,
        )
    ]
    ax = plotting.plot_sample_weights(
        voom_with_quality_weights(counts, design), ax=axes()
    )
    heights = [patch.get_height() for patch in artists(ax, "weights")]
    rows.append(
        row(
            "sample quality weights: bars",
            "voomWithQualityWeights(plot = TRUE)",
            scaled(heights, ref["weights"]),
            CLOSE,
        )
    )
    for key, values, options in (
        ("sa_robust", noisy, {"trend": True, "robust": True}),
        ("sa_plain", E, {}),
    ):
        fit = moderated_statistics(fit_linear_model(values, design), **options)
        ax = plotting.plot_residual_variances(fit, ax=axes())
        prior = artists(ax, "prior")[0].get_xydata()
        value = max(
            scaled(points(ax, "points", "outliers"), ref[f"{key} points"]),
            scaled(points(ax, "outliers"), ref[f"{key} outliers"]),
            scaled(prior if prior.shape[0] > 2 else prior[:, 1], ref[f"{key} prior"]),
        )
        label = (
            "a trend and robust moderation, 6 hypervariable genes"
            if options
            else "a constant prior"
        )
        n = len(ref[f"{key} outliers"])
        rows.append(
            row(
                f"residual variances, {label}: points, {n} outliers, prior",
                "plotSA",
                value,
                CLOSE,
            )
        )
    return rows


def check_tests(ref, table):
    ax = plotting.plot_md(table, ax=axes())
    value = max(
        scaled(points(ax, gid), ref[f"md {gid}"])
        for gid in ("up", "down", "not significant")
    )
    rows = [
        row(
            "MD plot: the genes up, down and not significant at 5% FDR",
            "plotMD(status = decideTests(fit))",
            value,
            EXACT,
        )
    ]
    ax = plotting.plot_volcano(table, ax=axes())
    value = scaled(points(ax, "up", "down", "not significant"), ref["volcano points"])
    labels = sorted(text.get_text() for text in artists(ax, "gene labels"))
    if labels != list(ref["volcano labels"]):
        value = np.inf
    rows.append(
        row(
            "volcano plot: points, and the 10 genes labelled",
            "volcanoplot(highlight = 10)",
            value,
            EXACT,
        )
    )
    ax = plotting.plot_p_values(table, ax=axes())
    heights = [patch.get_height() for patch in artists(ax, "p-values")]
    rows.append(
        row(
            "p-value histogram: tests in each bin of 0.05",
            "hist",
            bool(np.array_equal(heights, ref["histogram"])),
            0,
        )
    )
    return rows


def check_barcodes(ref, t, members, members_span):
    rows = []
    for key, statistics, chosen, span in (
        ("barcode", t, members, 0.45),
        ("barcode_span", t[:60], members_span, 1.0),
    ):
        ax = plotting.plot_barcode(statistics, chosen, span=span, ax=axes())
        bars = np.sort([s[0, 0] for s in artists(ax, "set")[0].get_segments()])
        value = max(
            scaled(bars, ref[f"{key} bars"]),
            scaled(artists(ax, "worm")[0].get_ydata(), ref[f"{key} worm"]),
        )
        label = f"barcode plot, {chosen.size} of {statistics.size} genes, span {span:g}"
        rows.append(
            row(f"{label}: bars and enrichment worm", "barcodeplot", value, EXACT)
        )
    return rows


def check_count_models(ref, counts, design, factors):
    """Drawn from R's own estimates, and for information from CytePool's fits."""
    rows = []
    fitted = fit_deseq2(counts, design).dispersion_estimates
    given = replace(
        fitted,
        **{
            name: ref[f"deseq2 {name}"]
            for name in ("base_mean", "genewise", "trended", "dispersion")
        },
        outlier=ref["deseq2 outlier"] == 1.0,
        all_zero=ref["deseq2 all_zero"] == 1.0,
    )
    for label, values in (
        ("drawn from DESeq2's estimates", given),
        ("of CytePool's fit of the counts", fitted),
    ):
        ax = plotting.plot_dispersions(values, ax=axes())
        value = max(
            scaled(points(ax, "gene-wise"), ref["deseq2 drawn gene-wise"]),
            scaled(points(ax, "final", "outliers"), ref["deseq2 drawn final"]),
            scaled(points(ax, "outliers"), ref["deseq2 drawn outliers"]),
            scaled(artists(ax, "fitted")[0].get_xydata(), ref["deseq2 drawn fitted"]),
        )
        n = len(ref["deseq2 drawn outliers"])
        rows.append(
            row(
                f"DESeq2's dispersions {label}: gene-wise, fitted, final, {n} outliers",
                "plotDispEsts",
                value,
                EXACT if values is given else None,
            )
        )
    fitted = fit_quasi_likelihood(counts, design, norm_factors=factors)
    with np.errstate(invalid="ignore"):
        quasi = ref["ql deviance"] / ref["ql df"]
    given = replace(
        fitted,
        covariate=ref["ql covariate"],
        quasi_dispersion=quasi,
        posterior_quasi_dispersion=ref["ql posterior"],
        prior_quasi_dispersion=ref["ql prior"],
    )
    for label, values in (
        ("drawn from edgeR's estimates", given),
        ("of CytePool's fit of the counts", fitted),
    ):
        ax = plotting.plot_quasi_dispersions(values, ax=axes())
        value = max(
            scaled(points(ax, "raw"), ref["ql drawn raw"]),
            scaled(points(ax, "squeezed"), ref["ql drawn squeezed"]),
            scaled(artists(ax, "prior")[0].get_xydata(), ref["ql drawn trend"]),
        )
        rows.append(
            row(
                f"edgeR's quasi-dispersions {label}: raw, squeezed, trend",
                "plotQLDisp",
                value,
                EXACT if values is given else None,
            )
        )
    drawn = ", ".join(f"{v:g}" for v in ref["ql drawn raw without df"])
    rows.append(
        (
            f"edgeR's quasi-dispersions of the {len(ref['ql drawn raw without df'])} "
            "genes without residual degrees of freedom",
            "plotQLDisp",
            "behaviour",
            f"edgeR draws {drawn}; CytePool, which has no quasi-dispersion for them, "
            "draws none",
            "n/a",
            "deliberate difference",
        )
    )
    return rows


def check_forests(ref, metas, features):
    rows = []
    for (method, test), meta in zip(FOREST_CASES, metas, strict=True):
        for feature in features:
            ax = plotting.plot_forest(meta, feature, ax=axes())
            segments = sorted(
                artists(ax, "intervals")[0].get_segments(), key=lambda s: -s[0, 1]
            )
            diamond = artists(ax, "pooled")[0].get_xy()[:3, 0]
            key = f"forest {method} {feature}"
            value = max(
                scaled([s[:, 0] for s in segments], ref[f"{key} intervals"]),
                scaled(diamond, ref[f"{key} pooled"]),
            )
            n = len(ref[f"{key} intervals"])
            rows.append(
                row(
                    f"forest plot, {method} and {test}, {feature} ({n} studies): "
                    "intervals, pooled estimate and interval",
                    "forest",
                    value,
                    CLOSE if method == "reml" else EXACT,
                )
            )
    return rows


def check_composition(ref, cube):
    ax = plotting.plot_composition(cube, by="condition", ax=axes())
    samples = [label.get_text() for label in ax.get_xticklabels()]
    heights = {
        (samples[i], container.get_label()): patch.get_height()
        for container in ax.containers
        for i, patch in enumerate(container)
    }
    keys = zip(ref["speckle bar_sample"], ref["speckle bar_cluster"], strict=True)
    value = scaled([heights[key] for key in keys], ref["speckle bar_height"])
    rows = [
        row(
            "composition: the segment of every sample and cell type",
            "plotCellTypeProps",
            value,
            EXACT,
        )
    ]
    design = Design(
        cube.samples,
        "~ condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim": "conditionstim"},
    )
    ax = plotting.plot_proportions(propeller(cube, design), by="condition", ax=axes())
    groups = cube.groups["group_id"].tolist()
    order = [list(ref["speckle table_clusters"]).index(group) for group in groups]
    means = [line.get_ydata()[0] for line in artists(ax, "means")]
    expected_means = np.concatenate(
        [ref[f"speckle mean_{level}"][order] for level in ("ctrl", "stim")]
    )
    clusters = list(ref["speckle clusters"])
    proportions = ref["speckle proportions"]
    expected = [
        proportions[clusters.index(group), s]
        for half in (range(6), range(6, 12))
        for group in groups
        for s in half
    ]
    drawn = np.concatenate([c.get_offsets()[:, 1] for c in artists(ax, "proportions")])
    value = max(scaled(means, expected_means), scaled(drawn, expected))
    texts = [text.get_text() for text in artists(ax, "adjusted p-values")]
    if texts != [f"adj. p {ref['speckle fdr'][i]:.2g}" for i in order]:
        value = np.inf
    rows.append(
        row(
            "proportions by condition: points, means and adjusted p-values",
            "propeller",
            value,
            CLOSE,
        )
    )
    return rows


def write_fixtures(ref, inputs):
    """The inputs and R's values, in four files, for the tests."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    every = ref | inputs
    files = {
        "count_models": ("deseq2", "ql"),
        "metafor": ("forest",),
        "speckle": ("speckle",),
    }
    for name, prefixes in files.items():
        arrays = {k: v for k, v in every.items() if k.split()[0] in prefixes}
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    taken = {prefix for prefixes in files.values() for prefix in prefixes}
    arrays = {k: v for k, v in every.items() if k.split()[0] not in taken}
    np.savez_compressed(FIXTURE_DIRECTORY / "limma.npz", **arrays)
    print(
        f"wrote limma.npz, count_models.npz, metafor.npz and speckle.npz to "
        f"{FIXTURE_DIRECTORY}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    rng = np.random.default_rng(20261101)
    counts = simulate(rng)
    genes = [f"g{j:03d}" for j in range(300)]
    samples = {
        "sample_id": [f"S{i}" for i in range(1, 9)],
        "condition": ["ctrl"] * 4 + ["stim"] * 4,
    }
    design = Design(
        samples,
        "~ condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim": "conditionstim"},
    )
    E = log2_cpm(counts)
    noise = rng.normal(0.0, 1.5, (8, 6))
    noisy = E.copy()
    noisy[:, 10:16] += noise
    n_cells = np.where(np.arange(8) < 6, 50, 5)[:, np.newaxis]
    cube = Cube(
        {"counts": counts[np.newaxis].astype(np.uint32)},
        samples=samples,
        groups={"group_id": ["T"]},
        features={"feature_id": genes},
        n_cells=n_cells,
    )
    table = limma_voom(
        from_bulk(counts, samples=samples, features=genes), design
    ).table("bulk", "stim")
    t = np.asarray(table["t"])
    members = np.unique(
        np.concatenate([np.argsort(-t)[:15:2], rng.choice(t.size, 17, replace=False)])
    )
    members_span = np.array([3, 7, 11, 20, 33, 41, 52])
    factors = tmm_factors(counts)
    limma = run_r(
        R_LIMMA,
        {
            "counts.bin": counts,
            "genes.txt": genes,
            "E.bin": E,
            "noisy.bin": noisy,
            "used.bin": n_cells[:, 0] >= 10,
            "t.bin": t,
            "set.bin": members + 1,
            "set_span.bin": members_span + 1,
            "norm_factors.bin": factors,
        },
        "RSCRIPT",
    )
    by_study = study_counts(rng, counts)
    results = study_results(by_study)
    metas = [
        meta_analysis(results, method=method, test=test)
        for method, test in FOREST_CASES
    ]
    features = forest_features(metas[0].table("bulk", "stim"))
    metafor = run_r(
        R_METAFOR, {"forest.json": forest_inputs(results, features)}, "METAFOR_RSCRIPT"
    )
    cells = composition(rng)
    composition_data = composition_cube(cells)
    labels = {"sample": [], "cluster": [], "group": []}
    for s, sample in enumerate(composition_data.samples["sample_id"].tolist()):
        for g, group in enumerate(composition_data.groups["group_id"].tolist()):
            labels["sample"] += [sample] * int(cells[s, g])
            labels["cluster"] += [group] * int(cells[s, g])
            labels["group"] += ["ctrl" if s < 6 else "stim"] * int(cells[s, g])
    speckle = run_r(R_SPECKLE, {"cells.json": labels}, "SPECKLE_RSCRIPT")

    ref = references(limma, metafor, speckle)
    print(
        "Reference: "
        + ", ".join(
            limma["versions"] + metafor["versions"][1:] + speckle["versions"][1:]
        )
    )
    rows = check_samples(ref, E, cube)
    rows += check_trends(ref, counts, E, noisy, design)
    rows += check_tests(ref, table)
    rows += check_barcodes(ref, t, members, members_span)
    rows += check_count_models(ref, counts, design, factors)
    rows += check_forests(ref, metas, features)
    rows += check_composition(ref, composition_data)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in rows:
        print("| " + " | ".join(line) + " |")
    failures = sum(line[-1] == "FAIL" for line in rows)
    print(f"{len(rows)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        inputs = {
            "counts": counts,
            "noise": noise,
            "t": t,
            "members": members,
            "members_span": members_span,
            "norm_factors": factors,
            "forest features": np.array(features),
            "speckle cells": cells,
        }
        inputs |= {f"forest counts {name}": values for name, values in by_study.items()}
        write_fixtures(ref, inputs)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
