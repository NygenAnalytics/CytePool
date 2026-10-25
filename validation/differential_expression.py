"""Check limma_voom and limma_trend against limma, group by group.

Simulates cells with a donor structure (seeds in ``main``), pools them with
``from_cells``, and runs ``limma_voom`` and ``limma_trend`` on every cell type. R runs
the workflow of the limma and edgeR User's Guides on the same pseudobulk counts: per
group, it leaves out the samples with fewer than 10 cells or no counts, keeps their
rows of the design ``model.matrix`` builds on the full sample table, and runs
``filterByExpr``, ``calcNormFactors``, ``voom`` (or ``cpm(log = TRUE, prior.count =
3)``), ``lmFit``, ``contrasts.fit``, ``eBayes`` and ``topTable(sort.by = "none")``,
one contrast at a time, so that a contrast limma refuses does not stop the others.
voom runs with ``adaptive.span = TRUE`` where ``limma_voom`` chooses the span from
the number of genes, as limma's voom does by default since limma 3.66, and with
``span = 0.5, adaptive.span = FALSE`` in the case that fixes it (``span=0.5``).

Cases: a paired design ``~ donor + condition`` (6 donors x 2 conditions, 4 cell types,
1,000 genes) by limma-voom and by limma-trend, and an unpaired design with a covariate,
``~ condition + age`` (10 donors, 3 cell types), by limma-voom with robust moderation
and a span fixed at 0.5, and by limma-trend without filtering or normalization
factors. Some sample x cell type
entries have fewer than 10 cells. In the NK cells both samples of donor D1, the
reference level, are left out, so the column ``donorD6`` is a combination of the
others; in the DC cells four samples remain, too few for residual degrees of freedom.

Per group, one row counts mismatches in the samples used, the features tested, the
coefficients limma cannot estimate, the contrasts tested by one program only (other
than the deliberate differences), voom's span and the residual and prior degrees of
freedom; one
gives the largest difference in logFC, AveExpr, t, P.Value, adj.P.Val and B over the
contrasts both compute, scaled by max(|reference|, 1), and for p-values by
max(p, 1e-300) max(1, |log p|). Tiers: exact (1e-10); close (1e-6) with robust
moderation, which limma solves only to a tolerance. limma-voom runs with
``contrast_standard_errors="limma"``, except in the robust case, whose contrasts are
single coefficients, for which the default exact standard errors are limma's.

Deliberate differences, in the NK cells: limma's ``contrasts.fit`` refuses
``D2_vs_D6`` (``donorD2 - donorD6``), which is estimable but weighs the aliased
column; CytePool computes it, and its values are checked against limma's values for
``D2_vs_D1`` (``donorD2``), which limma computes from the kept columns, where it
means D2 against D6. CytePool reports ``D2_vs_D1`` as not estimable.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages limma and edgeR, and jsonlite. Set
    ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/differential_expression.py
    PYTHONPATH=. python validation/differential_expression.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool.cube import from_cells
from cytepool.design import Design
from cytepool.differential_expression import limma_trend, limma_voom

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "differential_expression"
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
    "B": "b_statistic",
}

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, dir, name) writeBin(as.double(x), file.path(dir, name), size = 8)
# Runs expr without warnings; returns NULL or the error as "<step>: <message>". lmFit
# prints "Coefficients not estimable" to standard output (found by probing).
run <- function(expr) {
  capture.output(res <- tryCatch(suppressWarnings({ expr; NULL }),
                                 error = function(e) paste0(step, ": ", conditionMessage(e))))
  res
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  nf <- length(readLines(file.path(dir, "features.txt")))
  ng <- length(readLines(file.path(dir, "groups.txt")))
  levs <- jsonlite::fromJSON(file.path(dir, "levels.json"))
  # Factor columns are read as text, so that levels such as "01" keep their spelling.
  samples <- read.delim(file.path(dir, "samples.tsv"), check.names = FALSE,
                        colClasses = setNames(rep("character", length(levs)), names(levs)))
  for (v in names(levs)) samples[[v]] <- factor(samples[[v]], levels = levs[[v]])
  # One design for all samples, so that every group has the same columns.
  X_full <- model.matrix(as.formula(readLines(file.path(dir, "formula.txt"))), samples)
  cs <- read.delim(file.path(dir, "contrasts.tsv"), colClasses = "character")
  set <- jsonlite::fromJSON(file.path(dir, "settings.json"))
  counts <- array(rd(file.path(dir, "counts.bin")), dim = c(nf, nrow(samples), ng))
  n_cells <- matrix(rd(file.path(dir, "n_cells.bin")), nrow = nrow(samples), ncol = ng)
  for (g in seq_len(ng)) {
    out <- file.path(dir, paste0("group", g - 1))
    dir.create(out)
    y <- matrix(counts[, , g], nrow = nf)
    used <- n_cells[, g] >= set$min_cells & colSums(y) > 0
    X <- X_full[used, , drop = FALSE]
    wr(used, out, "used.bin")
    not_estimable <- character(0)
    # Found by probing: without residual df, voom and lmFit return values, eBayes stops.
    step <- "DGEList"
    status <- run({
      dge <- DGEList(y[, used, drop = FALSE])
      step <- "filterByExpr"
      keep <- if (set$filter) filterByExpr(dge, design = X) else rep(TRUE, nf)
      wr(keep, out, "keep.bin")
      dge <- dge[keep, , keep.lib.sizes = FALSE]
      step <- "calcNormFactors"
      if (set$normalization == "TMM") dge <- calcNormFactors(dge)
      if (set$method == "voom") {
        # adaptive.span is pinned either way, as its default changed in limma 3.66.
        # The result holds the span voom chose, and none when given one (found by
        # probing).
        step <- "voom"
        E <- if (is.null(set$span)) voom(dge, X, adaptive.span = TRUE) else
          voom(dge, X, span = set$span, adaptive.span = FALSE)
        wr(if (is.null(E$span)) set$span else E$span, out, "span.bin")
      } else {
        step <- "cpm"; E <- cpm(dge, log = TRUE, prior.count = set$prior_count)
      }
      step <- "lmFit"; fit <- lmFit(E, X)
      # Coefficients that lmFit cannot estimate are NA for every gene.
      not_estimable <- colnames(fit$coefficients)[colSums(!is.na(fit$coefficients)) == 0]
      wr(fit$df.residual, out, "df_residual.bin")
    })
    writeLines(if (is.null(status)) "fitted" else status, file.path(out, "status.txt"))
    writeLines(not_estimable, file.path(out, "not_estimable.txt"))
    # Found by probing: contrasts.fit stops for a contrast of an NA coefficient.
    if (is.null(status)) for (j in seq_len(nrow(cs))) {
      step <- "makeContrasts"
      error <- run({
        C <- makeContrasts(contrasts = cs$expression[j], levels = X)
        rownames(C) <- colnames(X)  # makeContrasts renames (Intercept)
        step <- "contrasts.fit"; cf <- contrasts.fit(fit, C)
        step <- "eBayes"
        eb <- eBayes(cf, trend = set$method == "trend", robust = set$robust)
        tt <- topTable(eb, coef = 1, number = Inf, sort.by = "none")
        wr(as.matrix(tt[, c("logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]),
           out, sprintf("contrast%d.bin", j - 1))
        wr(eb$df.prior, out, "df_prior.bin")
      })
      if (!is.null(error)) {
        writeLines(error, file.path(out, sprintf("contrast%d_error.txt", j - 1)))
      }
    }
  }
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR"))), file.path(root, "versions.txt"))
"""


def simulate_cells(rng, samples, cells, n_genes):
    """Cells x genes counts of every sample and cell type, with the cells' labels.

    Each cell type has its own log-normal expression profile. A donor shifts every gene
    of every cell type, stimulation changes a tenth of the genes of each cell type, and
    age a twentieth of them. Cells differ in depth (1,000 to 5,000 reads) and by gamma
    noise around their profile.
    """
    n = len(samples["sample_id"])
    donor = {d: rng.normal(0.0, 0.3, n_genes) for d in sorted(set(samples["donor"]))}
    age = np.asarray(samples.get("age", np.full(n, 50.0)), dtype=float)
    blocks, sample_labels, type_labels = [], [], []
    for cell_type, numbers in cells.items():
        profile = rng.normal(0.0, 1.8, n_genes)
        stim = np.where(rng.random(n_genes) < 0.1, rng.normal(0.0, 1.2, n_genes), 0.0)
        slope = np.where(rng.random(n_genes) < 0.05, rng.normal(0.0, 0.4, n_genes), 0.0)
        for s in range(n):
            log_mean = profile + donor[samples["donor"][s]] + slope * (age[s] - 50) / 25
            log_mean = log_mean + stim * (samples["condition"][s] == "stim")
            proportion = np.exp(log_mean) / np.exp(log_mean).sum()
            depth = rng.uniform(1000.0, 5000.0, numbers[s])[:, np.newaxis]
            rate = rng.gamma(4.0, 0.25, (numbers[s], n_genes)) * depth * proportion
            blocks.append(rng.poisson(rate))
            sample_labels += [samples["sample_id"][s]] * numbers[s]
            type_labels += [cell_type] * numbers[s]
    return np.vstack(blocks).astype(np.uint16), sample_labels, type_labels


def pooled(rng, samples, cells, n_genes):
    """The cube that from_cells pools from simulated cells, and the cells."""
    counts, labels, types = simulate_cells(rng, samples, cells, n_genes)
    cube = from_cells(
        counts, samples=labels, groups=types, sample_table=samples, layers="counts"
    )
    return cube, {"counts": counts, "samples": labels, "groups": types}


def paired_samples(n_donors):
    donors = [f"D{d}" for d in range(1, n_donors + 1)]
    return {
        "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
        "donor": [d for d in donors for _ in range(2)],
        "condition": ["ctrl", "stim"] * n_donors,
    }


def paired_design(cube, last, more=()):
    contrasts = {
        "stim_vs_ctrl": "conditionstim",
        "D2_vs_D1": "donorD2",
        f"D2_vs_{last}": f"donorD2 - donor{last}",
    } | dict(more)
    levels = {"condition": ["ctrl", "stim"]}
    return Design(
        cube.samples, "~ donor + condition", contrasts=contrasts, levels=levels
    )


def make_cases(rng):
    """Analyses by name: (cube, design, function, options, deliberate pairs)."""
    bounds = {"T": (40, 120), "B": (15, 60), "NK": (12, 40), "DC": (0, 9)}
    cells = {name: rng.integers(low, high, 12) for name, (low, high) in bounds.items()}
    cells["B"][[7, 10]] = [6, 0]  # D4_stim and D6_ctrl
    cells["NK"][[0, 1]] = [4, 7]  # both samples of D1, the reference donor
    cells["DC"][[2, 3, 4, 9]] = [12, 15, 11, 14]  # D2_ctrl, D2_stim, D3_ctrl, D5_stim
    cube, _ = pooled(rng, paired_samples(6), cells, 1000)
    # D3_vs_D2 weighs two columns that the NK cells keep.
    paired = paired_design(cube, "D6", [("D3_vs_D2", "donorD3 - donorD2")])
    # Without D1's samples, donorD6 = (Intercept) - donorD2 - ... - donorD5 in the NK
    # cells, and limma's estimate for donorD2 is that of donorD2 - donorD6.
    pairs = {("NK", "D2_vs_D6"): "D2_vs_D1"}
    donors = [f"U{d:02d}" for d in range(1, 11)]
    unpaired = {"sample_id": donors, "donor": donors, "condition": ["ctrl", "stim"] * 5}
    unpaired["age"] = rng.uniform(25.0, 70.0, 10).round(1).tolist()
    bounds = {"T": (30, 90), "B": (10, 50), "Mono": (10, 40)}
    cells = {name: rng.integers(low, high, 10) for name, (low, high) in bounds.items()}
    cells["B"][2] = 5
    cells["Mono"][[5, 8]] = [3, 8]
    other, _ = pooled(rng, unpaired, cells, 1000)
    contrasts = {"stim_vs_ctrl": "conditionstim", "age": "age"}
    with_age = (other, Design(other.samples, "~ condition + age", contrasts=contrasts))
    unfiltered = {"expression_filter": False, "normalization": None}
    robust, limma = {"robust": True, "span": 0.5}, {"contrast_standard_errors": "limma"}
    return {
        "paired, limma-voom": (cube, paired, limma_voom, limma, pairs),
        "paired, limma-trend": (cube, paired, limma_trend, {}, pairs),
        "unpaired, limma-voom, robust, span 0.5": (*with_age, limma_voom, robust, {}),
        "unpaired, limma-trend, unfiltered, unnormalized": (
            *with_age,
            limma_trend,
            unfiltered,
            {},
        ),
    }


def fixture_cases(rng):
    """The small study whose cells the tests pool, and its three analyses."""
    cells = {"T": rng.integers(15, 30, 8), "NK": rng.integers(12, 25, 8)}
    cells["T"][5] = 4  # D3_stim
    cells["NK"][[0, 1]] = [3, 5]  # both samples of D1
    samples = paired_samples(4)
    cube, inputs = pooled(rng, samples, cells, 120)
    inputs |= {f"sample_table {k}": np.array(v) for k, v in samples.items()}
    design, pairs = paired_design(cube, "D4"), {("NK", "D2_vs_D4"): "D2_vs_D1"}
    unfiltered = {"expression_filter": False, "normalization": None, "robust": True}
    limma = {"contrast_standard_errors": "limma"}
    return inputs, {
        "fixture voom": (cube, design, limma_voom, limma, pairs),
        "fixture robust": (
            cube,
            design,
            limma_voom,
            limma | {"robust": True, "span": 0.5},
            pairs,
        ),
        "fixture trend": (cube, design, limma_trend, unfiltered, pairs),
    }


def run_limma(cases):
    """Write every case for R, run limma, and read its outputs back."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, (cube, design, function, options, _) in cases.items():
            directory = root / names[name]
            directory.mkdir()
            columns = ["sample_id", *design.model_matrix.variables]
            rows = zip(*(design.samples[c].tolist() for c in columns), strict=True)
            contrasts = {name: _expression(design, name) for name in design.contrasts}
            normalized = options.get("normalization", "tmm") is not None
            settings = {
                "method": "voom" if function is limma_voom else "trend",
                "min_cells": 10,
                "filter": options.get("expression_filter", True),
                "normalization": "TMM" if normalized else "none",
                "robust": options.get("robust", False),
                "prior_count": 3,
                "span": options.get("span"),
            }
            levels = {k: list(v) for k, v in design.levels.items()}
            texts = {
                "groups.txt": cube.groups["group_id"],
                "features.txt": cube.features["feature_id"],
                "samples.tsv": ["\t".join(map(str, row)) for row in [columns, *rows]],
                "contrasts.tsv": [f"{k}\t{v}" for k, v in contrasts.items()],
                "formula.txt": [design.formula],
                "levels.json": [json.dumps(levels)],
                "settings.json": [json.dumps(settings)],
            }
            texts["contrasts.tsv"].insert(0, "name\texpression")
            for file, lines in texts.items():
                (directory / file).write_text("\n".join(lines) + "\n")
            arrays = {"counts.bin": cube.layer("counts"), "n_cells.bin": cube.n_cells.T}
            for file, values in arrays.items():
                values = np.ascontiguousarray(values, dtype="<f8")
                (directory / file).write_bytes(values.tobytes())
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        outputs = {
            name: [
                _read_group(root / names[name] / f"group{k}", design)
                for k in range(cube.n_groups)
            ]
            for name, (cube, design, *_) in cases.items()
        }
        return outputs, (root / "versions.txt").read_text().split("\n")[:3]


def _expression(design, name):
    """A contrast of the design as an expression over its columns."""
    weights = zip(design.contrasts[name], design.columns, strict=True)
    return " ".join(f"{w:+.17g}*{column}" for w, column in weights if w != 0.0)


def _read_group(directory, design):
    out = {
        "status": (directory / "status.txt").read_text().strip(),
        "not_estimable": (directory / "not_estimable.txt").read_text().split(),
        "contrasts": {},
    }
    for name in ("used", "keep", "df_residual", "df_prior", "span"):
        if (directory / f"{name}.bin").exists():
            out[name] = np.fromfile(directory / f"{name}.bin", dtype="<f8")
    for j, name in enumerate(design.contrasts):
        if (directory / f"contrast{j}_error.txt").exists():
            out["contrasts"][name] = (directory / f"contrast{j}_error.txt").read_text()
        elif (directory / f"contrast{j}.bin").exists():
            values = np.fromfile(directory / f"contrast{j}.bin", dtype="<f8")
            out["contrasts"][name] = values.reshape(6, -1)
    return out


def difference(ours, reference, p_value=False):
    """Largest |ours - reference| / scale, as the module docstring describes."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(ours - reference) / scale, initial=0.0))


def largest(table, reference):
    """The largest difference of a CytePool table from a topTable, over its columns."""
    return max(
        difference(table[column], reference[i], p_value="P" in name)
        for i, (name, column) in enumerate(COLUMNS.items())
    )


def compare(name, case, result, limma):
    """Rows of (check, reference, metric, value, tolerance, result) for one case."""
    cube, design, _, options, pairs = case
    tolerance = CLOSE if options.get("robust") else EXACT
    passed = {True: "pass", False: "FAIL"}
    rows = []
    for k, group in enumerate(cube.groups["group_id"]):
        theirs, record = limma[k], result.provenance["groups"][k]
        label = f"{name}, {group}"
        used = result.samples["used"][result.samples["group_id"] == group]
        if record["status"] != "fitted":
            # Both must refuse every contrast, for the same samples.
            errors = list(theirs["contrasts"].values()) or [theirs["status"]]
            refused = all(isinstance(e, str) for e in errors)
            same = refused and np.array_equal(used, theirs["used"].astype(bool))
            first = errors[0].strip() if refused else "values"
            value = f"limma: {first}; CytePool: {record['status']}"
            rows.append((label, "limma", "behaviour", value, "n/a", passed[same]))
            continue
        tables = {c: t for (g, c), t in result.tables.items() if g == group}
        ids = next(iter(tables.values()))["feature_id"]
        tested = np.isin(cube.features["feature_id"], ids)
        row = list(result.groups).index(group)
        df_prior = result.summary["prior_df"][row]
        prior = float(np.median(theirs["df_prior"]))
        # A contrast that one program tests and the other does not, other than the
        # deliberate differences.
        listed = {
            name
            for (where, contrast), meant in pairs.items()
            if where == group
            for name in (contrast, meant)
        }
        computed = {c for c, v in theirs["contrasts"].items() if not isinstance(v, str)}
        mismatches = (
            np.count_nonzero(used != theirs["used"].astype(bool))
            + np.count_nonzero(tested != theirs["keep"].astype(bool))
            + len(set(record["aliased_columns"]) ^ set(theirs["not_estimable"]))
            + len((set(tables) ^ computed) - listed)
            + (result.summary["df_residual"][row] != theirs["df_residual"][0])
            + (df_prior != prior and difference(df_prior, prior) > tolerance)
            # voom's span: limma's, or none for limma-trend.
            + (
                record.get("voom_span")
                != (theirs["span"][0] if "span" in theirs else None)
            )
        )
        rows.append(
            (
                f"{label}: samples, features, aliased columns, contrasts, span, df",
                "limma",
            )
            + ("mismatches", str(mismatches), "0", passed[mismatches == 0])
        )
        both = [c for c in tables if not isinstance(theirs["contrasts"][c], str)]
        value = max(
            (largest(tables[c], theirs["contrasts"][c]) for c in both), default=np.inf
        )
        rows.append(
            (f"{label}: {len(both)} contrast{'s' * (len(both) > 1)}", "topTable")
            + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
            + (passed[value <= tolerance],)
        )
        for (where, contrast), meant in pairs.items():
            if where == group:
                rows += deliberate(
                    label, contrast, meant, tables, theirs, record, tolerance
                )
    return rows


def deliberate(label, contrast, meant, tables, theirs, record, tolerance):
    """The rows of a group where limma refuses one contrast and misreads another.

    The first compares the contrast that limma refuses with limma's values for the one
    it misreads, which estimate the same quantity.
    """
    refused, misread = theirs["contrasts"][contrast], theirs["contrasts"][meant]
    expected = (
        isinstance(refused, str)
        and "non-estimable" in refused
        and not isinstance(misread, str)
        and meant in record["not_estimable"]
        and contrast in tables
    )
    value = largest(tables[contrast], misread) if expected else np.inf
    outcome = "deliberate difference" if expected else "FAIL"
    refusal = refused.strip() if isinstance(refused, str) else "values"
    return [
        (f"{label}: {contrast}, against limma's {meant}", "topTable")
        + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
        + ("pass" if value <= tolerance else "FAIL",),
        (f"{label}: {contrast} (estimable, weighs an aliased column)", "limma")
        + ("behaviour", f"limma: {refusal}; CytePool: values", "n/a", outcome),
        (f"{label}: {meant} (not estimable without donor D1)", "limma", "behaviour")
        + (f"limma: values, those of {contrast}; CytePool: not estimable", "n/a")
        + (outcome,),
    ]


def write_fixtures(inputs, cases, limma):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_DIRECTORY / "cells.npz", **inputs)
    arrays = {}
    for name, (cube, *_) in cases.items():
        for k, group in enumerate(cube.groups["group_id"]):
            out, key = limma[name][k], f"{name.removeprefix('fixture ')} {group}"
            arrays[f"{key} used"] = out["used"].astype(bool)
            arrays[f"{key} keep"] = out["keep"].astype(bool)
            arrays[f"{key} aliased"] = np.array(out["not_estimable"], dtype=str)
            arrays[f"{key} df_prior"] = out["df_prior"]
            if "span" in out:
                arrays[f"{key} span"] = out["span"]
            for contrast, values in out["contrasts"].items():
                if isinstance(values, str):
                    arrays[f"{key} {contrast} error"] = np.array(values.strip())
                else:
                    arrays[f"{key} {contrast}"] = values
    np.savez_compressed(FIXTURE_DIRECTORY / "limma.npz", **arrays)
    print(f"wrote cells.npz and limma.npz to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(20261025))
    inputs, fixtures = fixture_cases(np.random.default_rng(20261026))
    if arguments.write_fixtures:
        cases |= fixtures
    limma, versions = run_limma(cases)
    print("Reference: " + ", ".join(versions))
    print(
        "voom in R: adaptive.span = TRUE, or span = 0.5, adaptive.span = FALSE where "
        "a case fixes the span"
    )
    table = []
    for name, case in cases.items():
        cube, design, function, options, _ = case
        table += compare(name, case, function(cube, design, **options), limma[name])
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        write_fixtures(inputs, fixtures, {name: limma[name] for name in fixtures})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
