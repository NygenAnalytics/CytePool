"""Check the options of limma_voom and limma_trend for pseudobulk data against limma.

Simulates cells from 6 donors, each sampled before and after a stimulation (seeds in
``main``), pools them with ``from_cells`` and runs ``limma_voom`` and ``limma_trend``
on every cell type with the options for pseudobulk data, each against limma 3.62.1
and edgeR 4.4.0:

- sample quality weights (``quality_weights``): ``voomWithQualityWeights``, gene by
  gene and by REML, for limma-voom; ``arrayWeights`` on the log2 CPM, by REML and
  gene by gene, then ``lmFit`` with them as sample weights, for limma-trend;
- prior weights from cell numbers (``prior_weights="n_cells"``): R counts the cells of
  each sample and cell type from the cells' labels, not from the cube, and uses them
  as prior weights: ``voom(weights = n)``, whose weights times n enter ``lmFit``, as
  voom's help page says its weights modify prior weights; independently, edgeR's
  ``voomLmFit(prior.weights = n)``, which forms that product itself (found by
  probing); and ``lmFit(weights = n)`` for limma-trend. ``voomLmFit`` treats some
  genes with zero counts differently from voom and ``lmFit``: it lowers their residual
  df (its help page) and can move their average log count in the trend (found by
  probing). Here that happens only where ``min_counts`` leaves a donor with one
  sample, so it is compared in the analysis without ``min_counts``;
- a minimum total count per sample (``min_counts``): R leaves out the samples whose
  column sum is below it, besides those with fewer than 10 cells or no counts;
- p-values adjusted across cell types (``adjust_across_groups``): ``p.adjust(method =
  "BH")`` of the ``P.Value`` of every cell type stacked, one contrast at a time.

Otherwise R runs the workflow of ``validation/differential_expression.py``: per cell
type, the rows of the design ``model.matrix`` builds on the full sample table for
the samples kept, ``filterByExpr``, ``calcNormFactors``, voom with ``adaptive.span =
TRUE`` (or ``cpm(log = TRUE, prior.count = 3)``), ``lmFit``, ``contrasts.fit``,
``eBayes`` and ``topTable(sort.by = "none")``. The design is ``~ donor + condition``
with the contrasts ``stim_vs_ctrl`` and ``D3_vs_D2``. One sample is an outlier in
every cell type, and two are sequenced shallowly in one cell type each. In the NK
cells both samples of donor D1, the reference level, have fewer than 10 cells, so the
column ``donorD6`` is aliased. limma-voom runs with
``contrast_standard_errors="limma"``.

Per case and cell type, one row counts mismatches in the samples used, the features
tested, voom's span and the residual and prior degrees of freedom; one gives the
largest relative difference of the sample weights; one the largest difference in
logFC, AveExpr, t, P.Value, adj.P.Val and B over both contrasts, scaled by
max(|reference|, 1), and for p-values by max(p, 1e-300) max(1, |log p|), and with
prior weights in voom one more against ``voomLmFit``. With the adjustment across cell
types, one row per contrast compares it, scaled as p-values. Tiers: close (1e-6) for
everything computed from sample quality weights, which are iterative; exact (1e-10)
otherwise. A last row checks that both programs refuse prior weights together with
estimated sample weights: edgeR's ``voomLmFit`` and CytePool (limma's
``voomWithQualityWeights`` takes no prior weights).

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages limma and edgeR, and jsonlite. Set
    ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/differential_expression_options.py
    PYTHONPATH=. python validation/differential_expression_options.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool import Design, from_cells, limma_trend, limma_voom

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "differential_expression_options"
)
EXACT = 1e-10
CLOSE = 1e-6
# topTable's columns and CytePool's, in the same order.
COLUMNS = {
    "logFC": "log2_fold_change",
    "AveExpr": "average_log2_cpm",
    "t": "t",
    "P.Value": "p_value",
    "adj.P.Val": "adjusted_p_value",
    "B": "log_odds",
}
CONTRASTS = {"stim_vs_ctrl": "conditionstim", "D3_vs_D2": "donorD3 - donorD2"}
ACROSS = "adjusted_p_value_across_groups"
# The analyses the tests compare with limma, by the names the fixtures use.
FIXTURE_CASES = {
    "limma-voom, quality weights": "voom quality",
    "limma-voom, prior weights n_cells": "voom cells",
    "limma-trend, quality weights": "trend quality",
    "limma-trend, prior weights n_cells, min_counts": "trend cells",
}

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, dir, name) writeBin(as.double(x), file.path(dir, name), size = 8)
top <- function(fit, C, trend) {
  topTable(eBayes(contrasts.fit(fit, C), trend = trend), coef = 1, number = Inf,
           sort.by = "none")
}
columns <- c("logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")
samples <- read.delim(file.path(root, "samples.tsv"), colClasses = "character")
samples$condition <- factor(samples$condition, levels = c("ctrl", "stim"))
X_full <- model.matrix(~ donor + condition, samples)
cs <- read.delim(file.path(root, "contrasts.tsv"), colClasses = "character")
groups <- readLines(file.path(root, "groups.txt"))
ng <- length(groups)
nf <- length(readLines(file.path(root, "features.txt")))
counts <- array(rd(file.path(root, "counts.bin")), dim = c(nf, nrow(samples), ng))
# The cells of each sample and cell type, counted from the cells' labels.
n_cells <- unclass(table(
  factor(readLines(file.path(root, "cell_samples.txt")), levels = samples$sample_id),
  factor(readLines(file.path(root, "cell_groups.txt")), levels = groups)))
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  set <- jsonlite::fromJSON(file.path(dir, "settings.json"))
  p <- vector("list", ng * nrow(cs))
  dim(p) <- c(ng, nrow(cs))
  for (g in seq_len(ng)) {
    out <- file.path(dir, paste0("group", g - 1))
    dir.create(out)
    y <- matrix(counts[, , g], nrow = nf)
    used <- n_cells[, g] >= 10 & colSums(y) > 0 & colSums(y) >= set$min_counts
    X <- X_full[used, , drop = FALSE]
    n <- n_cells[used, g]  # the prior weights: the number of cells of each sample
    dge <- DGEList(y[, used, drop = FALSE])
    keep <- filterByExpr(dge, design = X)
    dge <- calcNormFactors(dge[keep, , keep.lib.sizes = FALSE])
    other <- NULL
    # lmFit prints the coefficients it cannot estimate; the output is not needed.
    printed <- capture.output({
      if (set$method == "voom" && set$weights == "quality") {
        v <- voomWithQualityWeights(dge, X, method = set$quality, adaptive.span = TRUE)
        E <- v$E
        W <- v$weights
        sw <- v$targets$sample.weights
      } else if (set$method == "voom") {
        v <- voom(dge, X, weights = n, adaptive.span = TRUE)
        E <- v$E
        # voom's weights modify its prior weights: the fit needs their product.
        W <- v$weights * matrix(n, nrow(v), ncol(v), byrow = TRUE)
        sw <- n
        # voomLmFit needs a design of full rank: the columns that qr() keeps.
        q <- qr(X)
        kept <- X[, sort(q$pivot[seq_len(q$rank)]), drop = FALSE]
        other <- voomLmFit(dge, kept, prior.weights = n, adaptive.span = TRUE)
        wr(other$df.residual, out, "voomlmfit_df.bin")
      } else {
        E <- cpm(dge, log = TRUE, prior.count = 3)
        sw <- W <- if (set$weights == "cells") n else arrayWeights(E, X, method = set$quality)
      }
      if (set$method == "voom") wr(v$span, out, "span.bin")
      fit <- lmFit(E, X, weights = W)
    })
    wr(sw, out, "weights.bin")
    wr(used, out, "used.bin")
    wr(keep, out, "keep.bin")
    wr(fit$df.residual[1], out, "df_residual.bin")
    for (j in seq_len(nrow(cs))) {
      C <- suppressWarnings(makeContrasts(contrasts = cs$expression[j], levels = X))
      rownames(C) <- colnames(X)  # makeContrasts renames (Intercept)
      tt <- top(fit, C, set$method == "trend")
      wr(as.matrix(tt[, columns]), out, sprintf("contrast%d.bin", j - 1))
      wr(eBayes(contrasts.fit(fit, C), trend = set$method == "trend")$df.prior, out,
         "df_prior.bin")
      p[[g, j]] <- tt$P.Value
      if (!is.null(other)) {
        C <- suppressWarnings(makeContrasts(contrasts = cs$expression[j], levels = kept))
        rownames(C) <- colnames(kept)
        wr(as.matrix(top(other, C, FALSE)[, columns]), out,
           sprintf("contrast%d_voomlmfit.bin", j - 1))
      }
    }
  }
  # The Benjamini-Hochberg adjustment of every cell type's p-values, stacked.
  for (j in seq_len(nrow(cs))) {
    adjusted <- p.adjust(unlist(p[, j]), method = "BH")
    group <- rep(seq_len(ng), lengths(p[, j]))
    for (g in seq_len(ng)) {
      wr(adjusted[group == g], file.path(dir, paste0("group", g - 1)),
         sprintf("contrast%d_across.bin", j - 1))
    }
  }
  if (isTRUE(set$both)) {
    y <- matrix(counts[, , 1], nrow = nf)
    error <- tryCatch({
      voomLmFit(y, X_full, prior.weights = n_cells[, 1], sample.weights = TRUE)
      "none"
    }, error = function(e) conditionMessage(e))
    writeLines(error, file.path(dir, "both.txt"))
  }
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR"))), file.path(root, "versions.txt"))
"""


def simulate_cells(rng, samples, cells, n_genes, outlier, shallow):
    """Cells x genes counts of every sample and cell type, with the cells' labels.

    Each cell type has its own log-normal expression profile. A donor shifts every gene
    of every cell type, and stimulation changes a tenth of the genes of each cell type.
    The ``outlier`` sample's profile is perturbed in every cell type (log-normal noise
    with sd 0.5), as an outlying donor's would be. Cells differ in depth, 1,000 to
    5,000 reads, or 100 to 300 in the samples that ``shallow`` lists for a cell type,
    and by gamma noise around their profile.
    """
    n = len(samples["sample_id"])
    donor = {d: rng.normal(0.0, 0.3, n_genes) for d in sorted(set(samples["donor"]))}
    blocks, sample_labels, type_labels = [], [], []
    for cell_type, numbers in cells.items():
        profile = rng.normal(0.0, 1.8, n_genes)
        stim = np.where(rng.random(n_genes) < 0.1, rng.normal(0.0, 1.2, n_genes), 0.0)
        noise = rng.normal(0.0, 0.5, n_genes)
        for s in range(n):
            log_mean = profile + donor[samples["donor"][s]] + noise * (s == outlier)
            log_mean = log_mean + stim * (samples["condition"][s] == "stim")
            proportion = np.exp(log_mean) / np.exp(log_mean).sum()
            low, high = (
                (100.0, 300.0) if s in shallow.get(cell_type, ()) else (1e3, 5e3)
            )
            depth = rng.uniform(low, high, numbers[s])[:, np.newaxis]
            rate = rng.gamma(4.0, 0.25, (numbers[s], n_genes)) * depth * proportion
            blocks.append(rng.poisson(rate))
            sample_labels += [samples["sample_id"][s]] * numbers[s]
            type_labels += [cell_type] * numbers[s]
    return np.vstack(blocks).astype(np.uint16), sample_labels, type_labels


def study(rng, n_donors, bounds, n_genes, outlier, shallow, few):
    """A cube pooled from simulated cells, its design, and the cells.

    ``few`` gives, per cell type, the samples with fewer than 10 cells.
    """
    donors = [f"D{d}" for d in range(1, n_donors + 1)]
    samples = {
        "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
        "donor": [d for d in donors for _ in range(2)],
        "condition": ["ctrl", "stim"] * n_donors,
    }
    cells = {
        name: rng.integers(low, high, 2 * n_donors) for name, (low, high) in bounds
    }
    for name, rows in few.items():
        cells[name][rows] = rng.integers(2, 10, len(rows))
    counts, labels, types = simulate_cells(
        rng, samples, cells, n_genes, outlier, shallow
    )
    cube = from_cells(
        counts,
        sample_labels=labels,
        group_labels=types,
        samples=samples,
        layers="counts",
    )
    design = Design(
        cube.samples,
        "~ donor + condition",
        contrasts=CONTRASTS,
        levels={"condition": ["ctrl", "stim"]},
    )
    inputs = {"counts": counts, "samples": labels, "groups": types}
    inputs |= {f"sample_table {k}": np.array(v) for k, v in samples.items()}
    return cube, design, inputs


def analyses(min_counts):
    """The analyses by name: function, options, and the settings of R's workflow."""
    limma, across = (
        {"contrast_standard_errors": "limma"},
        {"adjust_across_groups": True},
    )
    minimum = {"min_counts": min_counts}
    return {
        "limma-voom, quality weights": (
            limma_voom,
            limma | across | {"quality_weights": True},
            {"method": "voom", "weights": "quality", "quality": "genebygene"},
        ),
        "limma-voom, quality weights by REML, min_counts": (
            limma_voom,
            limma | minimum | {"quality_weights": {"method": "reml"}},
            {"method": "voom", "weights": "quality", "quality": "reml"},
        ),
        "limma-voom, prior weights n_cells": (
            limma_voom,
            limma | across | {"prior_weights": "n_cells"},
            {"method": "voom", "weights": "cells", "both": True},
        ),
        "limma-trend, quality weights": (
            limma_trend,
            across | {"quality_weights": True},
            {"method": "trend", "weights": "quality", "quality": "reml"},
        ),
        "limma-trend, quality weights gene by gene, min_counts": (
            limma_trend,
            minimum | {"quality_weights": {"method": "gene_by_gene"}},
            {"method": "trend", "weights": "quality", "quality": "genebygene"},
        ),
        "limma-trend, prior weights n_cells, min_counts": (
            limma_trend,
            across | minimum | {"prior_weights": "n_cells"},
            {"method": "trend", "weights": "cells"},
        ),
    }


def run_limma(cube, cells, design, cases):
    """Write the cube, its cells' labels and every case for R, run limma, and read
    its outputs back."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        columns = ["sample_id", "donor", "condition"]
        rows = zip(*(design.samples[c].tolist() for c in columns), strict=True)
        texts = {
            "groups.txt": cube.groups["group_id"],
            "features.txt": cube.features["feature_id"],
            "samples.tsv": ["\t".join(map(str, row)) for row in [columns, *rows]],
            "contrasts.tsv": ["name\texpression"]
            + [f"{k}\t{v}" for k, v in CONTRASTS.items()],
            "cell_samples.txt": cells["samples"],
            "cell_groups.txt": cells["groups"],
        }
        for file, lines in texts.items():
            (root / file).write_text("\n".join(lines) + "\n")
        counts = np.ascontiguousarray(cube.layer("counts"), dtype="<f8")
        (root / "counts.bin").write_bytes(counts.tobytes())
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, (_, options, settings) in cases.items():
            (root / names[name]).mkdir()
            settings = settings | {"min_counts": options.get("min_counts", 0)}
            (root / names[name] / "settings.json").write_text(json.dumps(settings))
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        outputs = {}
        for name in cases:
            directory = root / names[name]
            groups = []
            for k in range(cube.n_groups):
                found = {
                    path.stem: np.fromfile(path, dtype="<f8")
                    for path in (directory / f"group{k}").glob("*.bin")
                }
                for j, contrast in enumerate(CONTRASTS):
                    found[contrast] = found.pop(f"contrast{j}").reshape(6, -1)
                    found[f"{contrast} across"] = found.pop(f"contrast{j}_across")
                    if f"contrast{j}_voomlmfit" in found:
                        other = found.pop(f"contrast{j}_voomlmfit").reshape(6, -1)
                        found[f"{contrast} voomLmFit"] = other
                groups.append(found)
            both = directory / "both.txt"
            outputs[name] = (
                groups,
                both.read_text().strip() if both.exists() else None,
            )
        return outputs, (root / "versions.txt").read_text().split("\n")[:3]


def difference(ours, reference, p_value=False):
    """Largest |ours - reference| / scale, as the module docstring describes; inf
    when the shapes differ."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(ours - reference) / scale, initial=0.0))


def relative(ours, reference):
    """Largest |ours / reference - 1|; inf when there is nothing to compare or the
    shapes differ."""
    if ours is None:
        return np.inf
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    return float(np.max(np.abs(ours / reference - 1.0), initial=0.0))


def compare(name, cube, result, limma, options):
    """Rows of (check, reference, metric, value, tolerance, result) for one case."""
    tolerance = CLOSE if options.get("quality_weights") else EXACT
    passed = {True: "pass", False: "FAIL"}
    # Quality weights are limma's estimates; prior weights are R's count of cells.
    reference = "table"
    if options.get("quality_weights"):
        reference = "arrayWeights" if "trend" in name else "voomWithQualityWeights"
    rows = []
    for k, group in enumerate(result.groups):
        theirs, record = limma[k], result.provenance["groups"][k]
        label = f"{name}, {group}"
        if record["status"] != "fitted":  # every group of these studies is fitted
            rows.append(
                (label, "limma", "status", f"CytePool: {record['status']}")
                + ("fitted", "FAIL")
            )
            continue
        used = result.samples["used"][result.samples["group_id"] == group]
        tables = {c: result.table(group, c) for c in CONTRASTS}
        tested = np.isin(
            cube.features["feature_id"], tables["stim_vs_ctrl"]["feature_id"]
        )
        prior_df = result.summary["df_prior"][k]
        their_prior = float(np.median(theirs["df_prior"]))
        mismatches = (
            np.count_nonzero(used != theirs["used"].astype(bool))
            + np.count_nonzero(tested != theirs["keep"].astype(bool))
            + (result.summary["df_residual"][k] != theirs["df_residual"][0])
            + (prior_df != their_prior and difference(prior_df, their_prior) > tolerance)
            + (record.get("voom_span") != (theirs["span"][0] if "span" in theirs else None))
        )  # fmt: skip
        rows.append(
            (f"{label}: samples, features, span, df", "limma", "mismatches")
            + (str(mismatches), "0", passed[mismatches == 0])
        )
        weights = record.get("quality_weights") or record.get("prior_weights")
        value = relative(weights, theirs["weights"])
        rows.append(
            (f"{label}: sample weights", reference, "max relative difference")
            + (f"{value:.1e}", f"{tolerance:.0e}", passed[value <= tolerance])
        )
        for source, suffix in (("topTable", ""), ("voomLmFit", " voomLmFit")):
            if f"stim_vs_ctrl{suffix}" not in theirs:
                continue
            value = max(
                difference(tables[c][column], theirs[c + suffix][i], "P" in key)
                for c in CONTRASTS
                for i, (key, column) in enumerate(COLUMNS.items())
            )
            rows.append(
                (f"{label}: 2 contrasts", source, "max scaled difference")
                + (f"{value:.1e}", f"{tolerance:.0e}", passed[value <= tolerance])
            )
    if options.get("adjust_across_groups"):
        for contrast in CONTRASTS:
            ours = [
                result.tables[group, contrast][ACROSS]
                if (group, contrast) in result.tables
                else []
                for group in result.groups
            ]
            stacked = [limma[k][f"{contrast} across"] for k in range(len(limma))]
            value = difference(np.concatenate(ours), np.concatenate(stacked), True)
            rows.append(
                (f"{name}: {contrast} adjusted across cell types", "p.adjust")
                + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
                + (passed[value <= tolerance],)
            )
    return rows


def refusal(cube, design, theirs):
    """The row where both programs refuse prior weights with quality weights."""
    try:
        limma_voom(cube, design, prior_weights="n_cells", quality_weights=True)
        ours = "values"
    except ValueError as error:
        ours = "refuses" if "not both" in str(error) else str(error)
    both = "Can't specify prior.weights" in theirs and ours == "refuses"
    value = f"voomLmFit: {theirs}; CytePool: {ours}"
    return (
        "prior weights with estimated sample weights",
        "voomLmFit",
        "behaviour",
        value,
        "n/a",
        "pass" if both else "FAIL",
    )


def write_fixtures(inputs, min_counts, limma):
    """The cells, and limma's outputs of the tests' analyses (voomLmFit's, which
    agree with them, are left out to keep the files small)."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE_DIRECTORY / "cells.npz", **inputs, min_counts=np.array(min_counts)
    )
    arrays = {}
    for name, groups in limma.items():
        for k, out in enumerate(groups):
            for key, values in out.items():
                if not key.endswith("voomLmFit"):
                    arrays[f"{name} {k} {key}"] = values
    np.savez_compressed(FIXTURE_DIRECTORY / "limma.npz", **arrays)
    print(f"wrote cells.npz and limma.npz to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    # 6 donors, 3 cell types, 1,000 genes. D4_stim is the outlier; D2_stim (T) and
    # D5_ctrl (B) are shallow, with 28,210 and 9,202 counts, below min_counts, while
    # every other sample of 10 cells or more has at least 32,161.
    cube, design, cells = study(
        np.random.default_rng(20261027),
        6,
        [("T", (20, 150)), ("B", (12, 60)), ("NK", (12, 40))],
        1000,
        outlier=7,
        shallow={"T": [3], "B": [8]},
        few={"NK": [0, 1]},
    )
    cases = analyses(30_000)
    limma, versions = run_limma(cube, cells, design, cases)
    print("Reference: " + ", ".join(versions))
    table = []
    for name, (function, options, _) in cases.items():
        result = function(cube, design, **options)
        table += compare(name, cube, result, limma[name][0], options)
    both = next(text for _, text in limma.values() if text is not None)
    table.append(refusal(cube, design, both))
    # The small study of the tests: 4 donors, 2 cell types, 100 genes. D3_stim is
    # the outlier; D2_ctrl (T) is shallow, with 5,342 counts, the only sample of 10
    # cells or more below 10,000.
    fixture, fixture_design, inputs = study(
        np.random.default_rng(20261028),
        4,
        [("T", (15, 40)), ("NK", (12, 30))],
        100,
        outlier=5,
        shallow={"T": [2]},
        few={"NK": [0, 1]},
    )
    small = {k: v for k, v in analyses(10_000).items() if k in FIXTURE_CASES}
    if arguments.write_fixtures:
        theirs, _ = run_limma(fixture, inputs, fixture_design, small)
        for name, (function, options, _) in small.items():
            result = function(fixture, fixture_design, **options)
            rows = compare(f"fixture {name}", fixture, result, theirs[name][0], options)
            table += rows
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        groups = {FIXTURE_CASES[name]: theirs[name][0] for name in small}
        write_fixtures(inputs, 10_000, groups)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
