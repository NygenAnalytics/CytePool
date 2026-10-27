"""limma_voom on interferon-beta stimulated PBMCs (Kang et al. 2018), against limma.

Data: batch 2 of GEO series GSE96583 (Kang HM, Subramaniam M, Targ S, et al. 2018,
Nature Biotechnology 36, 89-94): peripheral blood mononuclear cells (PBMCs) of 8 lupus
patients, each split into a culture stimulated with interferon-beta for 6 hours and an
untreated one, sequenced with 10x Genomics, cells assigned to donors by genotype. The
script downloads three files into a directory outside the repository, checks their
SHA-256 and reads them as data, without extracting the archive. It stops if a fact it
relies on fails (MATRICES, N_GENES, TABLE_COLUMNS, CELL_TYPES): the shapes of the
counts, distinct gene ids, and the join of every cell to the cell table by condition
and barcode, never by position, where the 313 stimulated barcodes that are also
control barcodes end in an extra 1 in the table; every cell must match one row and
every row one cell.

from_cells pools the 24,673 singlets with a cell type into a cube of 16 samples (donor
x condition) x 8 cell types x 35,635 genes, keyed by Ensembl id with symbols; its
counts must equal sums of the cells computed with scipy. limma_voom tests the paired
design ~ donor + condition (the table's ind and stim), stimulated minus control, in
every cell type with its defaults. R runs the limma User's Guide workflow on the
counts, samples and cell numbers exported from the cube, with the result's min_cells
rule and R's defaults except where limma versions differ: filterByExpr with the
group's rows of the full design, calcNormFactors, voom with the span that the group's
record gives and adaptive.span = FALSE, lmFit, contrasts.fit, eBayes(legacy = TRUE)
and topTable(sort.by = "none"). limma_voom chooses that span from the number of genes
tested, as limma's voom does by default since limma 3.66, so R also computes
chooseLowessSpan(nrow(counts), small.n = 50, min.span = 0.3, power = 1/3), the span
voom's help page gives for adaptive.span = TRUE. The samples used, features tested,
aliased columns, degrees of freedom and span must be identical, and logFC, AveExpr, t,
P.Value, adj.P.Val and B agree to 1e-10, scaled by max(|reference|, 1), and p-values
by max(p, 1e-300) max(1, |log p|): the exact tier, as without robust moderation every
step is closed-form or iterates to rounding error.

Information rows, per cell type: samples used and left out, features tested, genes
called at a 5% FDR; the Hallmark interferon-alpha response genes (MSigDB; Liberzon et
al. 2015, Cell Systems 1, 417-425) called up and down, and their median logFC; and a
negative control without R, which should make few calls: the control samples that the
cell type's fit used, split into two halves in every way, each split once, and fitted
with ~ split. Also printed: the time of limma_voom (cube to tables) and of R's workflow
(counts to tables), peak memory, the machine and the versions. With --write-fixtures,
the script writes the megakaryocytes' counts of the Hallmark genes and of 300 tested
genes, the samples, and limma's results on exactly that subset.

Requirements:
    Python 3.13 with numpy, numba and scipy (for Matrix Market files), run from the
    repository root so that cytepool imports from the working tree, and access to
    ftp.ncbi.nlm.nih.gov for the first run (77 MB). R 4.4 or newer with limma, edgeR
    and jsonlite; set RSCRIPT to the Rscript executable when it is not on PATH. The
    data directory is --data, else GSE96583_DATA, else cytepool-validation/GSE96583
    under XDG_CACHE_HOME or ~/.cache; it must be outside the repository.

Usage:
    PYTHONPATH=. python validation/interferon_stimulated_pbmcs.py
    PYTHONPATH=. python validation/interferon_stimulated_pbmcs.py --write-fixtures
"""

import argparse
import datetime
import gzip
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numba
import numpy as np
import scipy
import scipy.io
import scipy.sparse

import cytepool
from cytepool import Design, from_cells, limma_voom

SOURCE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/"
FILES = {  # bytes and SHA-256, as downloaded on 2026-10-25
    "GSE96583_RAW.tar": (
        76_195_840,
        "e5d41a3248a813f99d68fd5c9eb9773de7f46a83680a67f4a02d683b8955fe80",
    ),
    "GSE96583_batch2.genes.tsv.gz": (
        277_054,
        "93aa4e9b530ef9d6411ca129b416324c5cc1cc5a01a1fa6ed4f4a845480ed3ca",
    ),
    "GSE96583_batch2.total.tsne.df.tsv.gz": (
        756_342,
        "1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09",
    ),
}
# Per condition: the counts (genes x cells) and the barcodes in the archive, and the
# numbers of cells and of stored counts.
MATRICES = {
    "ctrl": ("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz", 14_619, 8_732_747),
    "stim": ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz", 14_446, 8_838_098),
}
N_GENES = 35_635
TABLE_COLUMNS = ["tsne1", "tsne2", "ind", "stim", "cluster", "cell", "multiplets"]
CELL_TYPES = {
    "CD4 T cells": 10_389,
    "CD14+ Monocytes": 5_385,
    "B cells": 2_573,
    "CD8 T cells": 2_042,
    "NK cells": 1_993,
    "FCGR3A+ Monocytes": 1_599,
    "Dendritic cells": 432,
    "Megakaryocytes": 260,
}
# MSigDB's Hallmark gene set HALLMARK_INTERFERON_ALPHA_RESPONSE (M5911, 97 genes, MSigDB
# v2026.2.Hs), "genes up-regulated in response to alpha interferon proteins": Liberzon
# A, Birger C, Thorvaldsdottir H, Ghandi M, Mesirov JP, Tamayo P (2015). The Molecular
# Signatures Database hallmark gene set collection. Cell Systems 1, 417-425.
# https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE
# Copyright (c) 2004-2026 Broad Institute, Inc., Massachusetts Institute of Technology,
# and Regents of the University of California, under the Creative Commons Attribution
# 4.0 International License (CC BY 4.0).
INTERFERON_ALPHA_RESPONSE = (
    "ADAR B2M BATF2 BST2 C1S CASP1 CASP8 CCRL2 CD47 CD74 CMPK2 CMTR1 CNP CSF1 CXCL10 "
    "CXCL11 DDX60 DHX58 EIF2AK2 ELF1 EPSTI1 GBP2 GBP4 GMPR HELZ2 HERC6 HLA-C IFI27 "
    "IFI30 IFI35 IFI44 IFI44L IFIH1 IFIT2 IFIT3 IFITM1 IFITM2 IFITM3 IL15 IL4R IL7 "
    "IRF1 IRF2 IRF7 IRF9 ISG15 ISG20 LAMP3 LAP3 LGALS3BP LPAR6 LY6E MOV10 MVB12A MX1 "
    "NCOA7 NMI NUB1 OAS1 OASL OGFR PARP12 PARP14 PARP9 PLSCR1 PNPT1 PROCR PSMA3 PSMB8 "
    "PSMB9 PSME1 PSME2 RIPK2 RNF31 RSAD2 RTP4 SAMD9 SAMD9L SELL SLC25A28 SP110 STAT2 "
    "TAP1 TDRD7 TENT5A TMEM140 TRAFD1 TRIM14 TRIM21 TRIM25 TRIM26 TRIM5 TXNIP UBA7 "
    "UBE2L6 USP18 WARS1"
).split()
# Two members have earlier symbols in the data's annotation; MSigDB's page of the set
# lists those as the members' source ids.
EARLIER_SYMBOLS = {"TENT5A": "FAM46A", "WARS1": "WARS"}
Y_GENES = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY"]
FDR, EXACT = 0.05, 1e-10
FIXTURE_GROUP = "Megakaryocytes"
FIXTURE = Path(__file__).resolve().parents[1] / "tests/data/interferon_stimulated_pbmcs"
# topTable's columns and CytePool's, in the same order.
COLUMNS = {
    "logFC": "log2_fold_change",
    "AveExpr": "average_log2_cpm",
    "t": "t",
    "P.Value": "p_value",
    "adj.P.Val": "adjusted_p_value",
    "B": "log_odds",
}

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, dir, name) writeBin(as.double(x), file.path(dir, name), size = 8)
# voom and lmFit print "Coefficients not estimable" and warn "Partial NA coefficients"
# for a column of the design that is zero in a group (found by probing); the script
# compares those columns with CytePool's.
quietly <- function(expr) withCallingHandlers(capture.output(expr),
  warning = function(w)
    if (startsWith(conditionMessage(w), "Partial NA")) invokeRestart("muffleWarning"))
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  set <- jsonlite::fromJSON(file.path(dir, "settings.json"))
  # Text columns, so that donor ids such as "101" stay labels.
  samples <- read.delim(file.path(dir, "samples.tsv"), colClasses = "character")
  for (v in names(set$levels)) samples[[v]] <- factor(samples[[v]], set$levels[[v]])
  X_full <- model.matrix(as.formula(set$formula), samples)
  stopifnot(identical(colnames(X_full), set$columns))
  ns <- nrow(samples); ng <- length(set$spans)
  counts <- array(rd(file.path(dir, "counts.bin")), dim = c(set$n_features, ns, ng))
  n_cells <- matrix(rd(file.path(dir, "n_cells.bin")), nrow = ns)
  C <- matrix(set$contrast, dimnames = list(colnames(X_full), "contrast"))
  out <- vector("list", ng)
  start <- proc.time()[["elapsed"]]
  for (g in seq_len(ng)) {
    y <- matrix(counts[, , g], ncol = ns)
    used <- n_cells[, g] >= set$min_cells & colSums(y) > 0
    X <- X_full[used, , drop = FALSE]
    dge <- DGEList(y[, used, drop = FALSE])
    keep <- filterByExpr(dge, design = X)
    dge <- calcNormFactors(dge[keep, , keep.lib.sizes = FALSE], method = "TMM")
    quietly({v <- voom(dge, X, span = set$spans[g], adaptive.span = FALSE)
             fit <- lmFit(v, X)})
    eb <- eBayes(contrasts.fit(fit, C), legacy = TRUE)
    tt <- topTable(eb, coef = 1, number = Inf, sort.by = "none")
    # The span voom chooses with adaptive.span = TRUE (voom's help page).
    rule <- chooseLowessSpan(nrow(dge), small.n = 50, min.span = 0.3, power = 1 / 3)
    out[[g]] <- list(used = used, keep = keep, df = c(fit$df.residual[1], eb$df.prior),
                     span = rule,
                     table = as.matrix(tt[, c("logFC", "AveExpr", "t", "P.Value",
                                              "adj.P.Val", "B")]),
                     aliased = colnames(X)[colSums(!is.na(fit$coefficients)) == 0])
  }
  writeLines(format(proc.time()[["elapsed"]] - start), file.path(dir, "seconds.txt"))
  cat(sprintf("R %s, limma %s, edgeR %s, %s: voom with span = %s, %s\n", getRversion(),
              packageVersion("limma"), packageVersion("edgeR"), case,
              paste(unique(set$spans), collapse = ", "), "adaptive.span = FALSE"))
  for (g in seq_len(ng)) {
    for (part in c("used", "keep", "df", "span", "table"))
      wr(out[[g]][[part]], dir, sprintf("group%d_%s.bin", g - 1, part))
    writeLines(out[[g]]$aliased, file.path(dir, sprintf("group%d_aliased.txt", g - 1)))
  }
}
# The peak resident memory of this process (Linux).
status <- if (file.exists("/proc/self/status")) readLines("/proc/self/status")
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR")),
             grep("^VmHWM", status, value = TRUE)), file.path(root, "versions.txt"))
"""


def require(condition, fact):
    """Stop with a message when a fact about the data does not hold."""
    if not condition:
        sys.exit(f"GSE96583 does not match what this script expects: {fact}")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(directory):
    """Download the files missing from directory, keeping a download only if its
    SHA-256 is right, and check the SHA-256 of the files already there."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in FILES.items():
        path = directory / name
        if path.exists():
            found = sha256(path)
            message = f"{path} has SHA-256 {found}, not {checksum}; delete it"
            require(found == checksum, message + " to download it again")
            continue
        print(f"downloading {SOURCE}{name} ({size:,} bytes) to {directory}", flush=True)
        partial = directory / f"{name}.part"
        with urllib.request.urlopen(SOURCE + name, timeout=120) as response:
            with partial.open("wb") as file:
                shutil.copyfileobj(response, file, 1 << 20)
        found = sha256(partial)
        if found != checksum:
            partial.unlink()
        require(found == checksum, f"the download of {name} has SHA-256 {found}")
        partial.replace(path)


def read_cells(directory):
    """The singlets that have a cell type: cells x genes counts, labels and genes."""
    with gzip.open(directory / "GSE96583_batch2.genes.tsv.gz", "rt") as text:
        genes = [line.rstrip("\n").split("\t") for line in text]
    require(
        {len(gene) for gene in genes} == {2}
        and len(genes) == len({gene[0] for gene in genes}) == N_GENES,
        f"the genes file has {N_GENES:,} rows: a distinct Ensembl id and a symbol",
    )
    with gzip.open(directory / "GSE96583_batch2.total.tsne.df.tsv.gz", "rt") as text:
        header, *lines = text.read().splitlines()
    table = {name: rest for name, *rest in (line.split("\t") for line in lines)}
    require(
        header.split("\t") == TABLE_COLUMNS
        and len(table) == len(lines)
        and {len(rest) for rest in table.values()} == {7},
        f"the cell table has a row per cell, named by barcode, of {TABLE_COLUMNS}",
    )
    blocks, matched, labels = [], Counter(), {"sample": [], "group": []}
    with tarfile.open(directory / "GSE96583_RAW.tar") as archive:
        barcodes = {}
        for condition, (_, member, n_cells, _) in MATRICES.items():
            with archive.extractfile(member) as raw:
                barcodes[condition] = gzip.decompress(raw.read()).decode().split()
            require(
                len(set(barcodes[condition])) == len(barcodes[condition]) == n_cells,
                f"{member} lists {n_cells:,} distinct barcodes",
            )
        shared = set(barcodes["ctrl"]) & set(barcodes["stim"])
        require(len(shared) == 313, "313 stimulated barcodes are also control ones")
        for condition, (member, _, n_cells, stored) in MATRICES.items():
            with archive.extractfile(member) as raw, gzip.open(raw) as stream:
                try:
                    counts = scipy.io.mmread(stream, spmatrix=False)
                except TypeError:  # scipy without the keyword returns a coo_matrix
                    counts = scipy.io.mmread(stream)
            require(
                counts.shape == (N_GENES, n_cells) and counts.nnz == stored,
                f"{member} has {N_GENES:,} x {n_cells:,} counts, {stored:,} stored",
            )
            # The table adds a 1 to a stimulated barcode that is also a control one.
            names = [
                f"{barcode}1" if condition == "stim" and barcode in shared else barcode
                for barcode in barcodes[condition]
            ]
            matched.update(names)
            cells = [table.get(name) for name in names]
            require(
                all(cell is not None and cell[3] == condition for cell in cells),
                f"every {condition} barcode names a {condition} cell of the table",
            )
            kept = [
                position
                for position, cell in enumerate(cells)
                if cell[6] == "singlet" and cell[5] != "NA"
            ]
            blocks.append(counts.T.tocsr()[kept])
            labels["sample"] += [f"{cells[p][2]}_{condition}" for p in kept]
            labels["group"] += [cells[p][5] for p in kept]
    require(
        set(matched) == set(table) and set(matched.values()) == {1},
        "every cell of the table matches one barcode, and every barcode one cell",
    )
    require(Counter(labels["group"]) == CELL_TYPES, f"the singlets are {CELL_TYPES}")
    return scipy.sparse.vstack(blocks).tocsr(), labels, genes


def make_cube(cells, labels, genes):
    """The cube of the cells: samples are donor x condition, groups cell types."""
    donors = sorted({sample.split("_")[0] for sample in labels["sample"]})
    require(len(donors) == 8, "the singlets come from 8 donors")
    samples = {
        "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
        "donor": [d for d in donors for _ in range(2)],
        "condition": ["ctrl", "stim"] * len(donors),
    }
    return from_cells(
        cells,
        sample_labels=labels["sample"],
        group_labels=labels["group"],
        samples=samples,
        features={"feature_id": [g[0] for g in genes], "symbol": [g[1] for g in genes]},
        layers="counts",
    )


def pooling_differences(cube, cells, labels):
    """How many counts of the cube differ from sums of its cells, which a scipy
    product with a sample-and-group indicator matrix computes."""
    sample = {s: i for i, s in enumerate(cube.samples["sample_id"].tolist())}
    group = {g: i for i, g in enumerate(cube.groups["group_id"].tolist())}
    rows = [
        group[g] * cube.n_samples + sample[s]
        for s, g in zip(labels["sample"], labels["group"], strict=True)
    ]
    shape = (cube.n_groups * cube.n_samples, len(rows))
    columns = np.arange(len(rows))
    indicator = scipy.sparse.csr_array((np.ones(len(rows)), (rows, columns)), shape)
    sums = (indicator @ cells).toarray().reshape(cube.n_groups, cube.n_samples, -1)
    return int(np.count_nonzero(sums != cube.layer("counts")))


def paired_design(cube):
    """~ donor + condition: each donor's stimulated sample against its control."""
    return Design(
        cube.samples,
        "~ donor + condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )


def voom_span(result, group):
    """The LOWESS span of voom's trend in a group, as the group's record gives it."""
    record = next(r for r in result.provenance["groups"] if r["group_id"] == group)
    if record["voom_span"] is None:
        raise ValueError(f"voom did not run in group {group!r}: {record['reason']}")
    return float(record["voom_span"])


def run_limma(cases):
    """Run the R workflow on each case, {label: (cube, design, result)}, with the
    min_cells and the voom span of every group of the CytePool result; return R's
    outputs and seconds per case, the versions, and R's peak resident memory in
    MiB."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for k, (cube, design, result) in enumerate(cases.values()):
            directory = root / f"case{k}"
            directory.mkdir()
            # The counts and cell numbers go to R in the cube's sample order.
            ids = design.samples["sample_id"].tolist()
            if ids != cube.samples["sample_id"].tolist():
                raise ValueError("the design must list the cube's samples, in order")
            columns = ["sample_id", *design.model_matrix.variables]
            rows = zip(*(design.samples[column].tolist() for column in columns))
            lines = ["\t".join(map(str, row)) for row in [columns, *rows]]
            (directory / "samples.tsv").write_text("\n".join(lines) + "\n")
            settings = {
                "formula": design.formula,
                "levels": {name: list(v) for name, v in design.levels.items()},
                "columns": list(result.contrasts.columns),
                "contrast": result.contrasts.matrix[:, 0].tolist(),
                "min_cells": result.settings["min_cells"],
                "spans": [voom_span(result, group) for group in result.groups],
                "n_features": cube.n_features,
            }
            (directory / "settings.json").write_text(json.dumps(settings))
            arrays = {"counts.bin": cube.layer("counts"), "n_cells.bin": cube.n_cells.T}
            for file, values in arrays.items():
                values = np.ascontiguousarray(values, dtype="<f8")
                (directory / file).write_bytes(values.tobytes())
        (root / "cases.txt").write_text(
            "".join(f"case{k}\n" for k in range(len(cases)))
        )
        (root / "workflow.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        sys.stdout.flush()  # R's lines after this program's
        command = [rscript, "--vanilla", str(root / "workflow.R"), str(root)]
        subprocess.run(command, check=True)
        outputs = {}
        for k, (label, (cube, *_)) in enumerate(cases.items()):
            directory, groups = root / f"case{k}", []
            for g in range(cube.n_groups):
                read = {
                    name: np.fromfile(directory / f"group{g}_{name}.bin", dtype="<f8")
                    for name in ("used", "keep", "df", "span", "table")
                }
                aliased = (directory / f"group{g}_aliased.txt").read_text().split()
                read |= {"used": read["used"] == 1.0, "keep": read["keep"] == 1.0}
                read |= {"table": read["table"].reshape(6, -1), "aliased": aliased}
                groups.append(read)
            outputs[label] = groups, float((directory / "seconds.txt").read_text())
        versions = (root / "versions.txt").read_text().split("\n")
        peak = int(versions[3].split()[1]) / 1024 if versions[3] else np.nan
        return outputs, versions[:3], peak


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


def compare(label, cube, result, limma):
    """Rows of (check, reference, metric, value, tolerance, result) per group."""
    rows, verdict = [], {True: "pass", False: "FAIL"}
    for k, group in enumerate(result.groups):
        theirs, record = limma[k], result.provenance["groups"][k]
        if record["status"] != "fitted":
            rows.append((label + group, "limma", "status", record["status"], "fitted"))
            rows[-1] += ("FAIL",)
            continue
        table = result.table(group, "stim_vs_ctrl")
        used = result.samples["used"][result.samples["group_id"] == group]
        tested = np.isin(cube.features["feature_id"], table["feature_id"])
        # Equal, also when both are infinite, or close; a NaN is a mismatch.
        prior, limma_prior = result.summary["df_prior"][k], theirs["df"][1]
        same_prior = prior == limma_prior or difference(prior, limma_prior) <= EXACT
        span = voom_span(result, group)
        mismatches = (
            np.count_nonzero(used != theirs["used"])
            + np.count_nonzero(tested != theirs["keep"])
            + (record["aliased_columns"] != theirs["aliased"])
            + (result.summary["df_residual"][k] != theirs["df"][0])
            + (not same_prior)
            + (not difference(span, theirs["span"][0]) <= EXACT)
        )
        check = f"{label}{group}: samples, features, aliased columns, df, span"
        rows.append((check, "limma", "mismatches", str(mismatches), "0"))
        rows[-1] += (verdict[mismatches == 0],)
        value = max(
            difference(
                table[column], theirs["table"][i], name in ("P.Value", "adj.P.Val")
            )
            for i, (name, column) in enumerate(COLUMNS.items())
        )
        check, metric = f"{label}{group}: {', '.join(COLUMNS)}", "max scaled difference"
        rows.append((check, "topTable", metric, f"{value:.1e}", f"{EXACT:.0e}"))
        rows[-1] += (verdict[value <= EXACT],)
    return rows


def hallmark_features(cube):
    """Whether each feature of the cube is a gene of the Hallmark set."""
    symbols = [EARLIER_SYMBOLS.get(s, s) for s in INTERFERON_ALPHA_RESPONSE]
    found = set(symbols) & set(cube.features["symbol"].tolist())
    require(len(found) == 97, "the genes file has the 97 genes of the Hallmark set")
    return np.isin(cube.features["symbol"], symbols)


def information(cube, result, splits, calls):
    """Rows of information per group: samples, features and calls; the Hallmark
    genes; the negative control."""
    rows, hallmark, summary = [], hallmark_features(cube), result.summary
    for k, group in enumerate(result.groups):
        table = result.table(group, "stim_vs_ctrl")
        called, change = table["adjusted_p_value"] < FDR, table["log2_fold_change"]
        up, down = called & (change > 0), called & (change < 0)
        listed = np.isin(table["feature_id"], cube.features["feature_id"][hallmark])
        symbols = table["symbol"][listed]
        require(len(set(symbols)) == len(symbols), "one tested feature per gene")
        numbers = [len(found) for found in calls[group]]
        used, left_out = summary["n_samples_used"][k], summary["n_samples_left_out"][k]
        sizes = " vs ".join(str(len(donors)) for donors in splits[group][0])
        rows += [
            (
                f"{group}: samples, features, calls",
                "none",
                "samples used, left out; features tested; up, down at 5% FDR",
                f"{used}, {left_out}; {len(table):,}; {up.sum():,}, {down.sum():,}",
            ),
            (
                f"{group}: Hallmark interferon-alpha response",
                "MSigDB",
                "up, down (genes) at 5% FDR; tested; median logFC",
                f"{(up & listed).sum()}, {(down & listed).sum()} "
                f"({' '.join(table['symbol'][down & listed]) or 'none'}); "
                f"{listed.sum()}; {np.median(change[listed]):.2f}",
            ),
            (
                f"{group}: negative control, splits of the control samples",
                "none",
                "splits (group sizes); splits with calls; calls; most in a split",
                f"{len(numbers)} ({sizes}); {np.count_nonzero(numbers)}; "
                f"{sum(numbers)}; {max(numbers)}",
            ),
        ]
    return [row + ("n/a", "information") for row in rows]


def halves(items):
    """Every split of items into groups of n // 2 and n - n // 2, each split once, as
    the first group; it holds the first item when n is even."""
    n = len(items)
    if n % 2:
        return [list(chosen) for chosen in itertools.combinations(items, n // 2)]
    rest = itertools.combinations(items[1:], n // 2 - 1)
    return [[items[0], *chosen] for chosen in rest]


def negative_control(cube, result):
    """For every group, the splits of the control samples that result used in it into
    halves, as the donors of groups A and B, and the symbols called at 5% FDR in each
    split, smallest p-value first."""
    splits, calls, samples = {}, {}, result.samples
    for group in result.groups:
        used = samples["sample_id"][(samples["group_id"] == group) & samples["used"]]
        chosen = np.isin(cube.samples["sample_id"], used)
        controls = cube.samples.filter(chosen & (cube.samples["condition"] == "ctrl"))
        donors = controls["donor"].tolist()
        splits[group] = [
            (first, [donor for donor in donors if donor not in first])
            for first in halves(donors)
        ]
        calls[group] = []
        for first, _ in splits[group]:
            split = ["A" if donor in first else "B" for donor in donors]
            design = Design(
                controls.with_columns(split=split),
                "~ split",
                contrasts={"B_vs_A": "splitB"},
            )
            table = limma_voom(cube, design, groups=group).table(group, "B_vs_A")
            found = np.flatnonzero(table["adjusted_p_value"] < FDR)
            order = found[np.argsort(table["p_value"][found], kind="stable")]
            calls[group].append(table["symbol"][order].tolist())
    return splits, calls


def fixture_case(cube, result):
    """One cell type of the cube, with the Hallmark genes and 300 other genes drawn
    at random from those that result tested in it; its design and its result."""
    hallmark = hallmark_features(cube)
    table = result.table(FIXTURE_GROUP, "stim_vs_ctrl")
    tested = np.isin(cube.features["feature_id"], table["feature_id"])
    rng = np.random.default_rng(20261025)
    others = rng.choice(np.flatnonzero(tested & ~hallmark), 300, replace=False)
    chosen = np.sort(np.concatenate([np.flatnonzero(hallmark), others]))
    subset = cube.select(groups=FIXTURE_GROUP, features=chosen)
    design = paired_design(subset)
    return subset, design, limma_voom(subset, design)


def write_fixtures(case, limma):
    """Write the fixture's counts, cells, samples, features and limma's results."""
    cube, group = case[0], limma[0]
    arrays = {
        "group_id": FIXTURE_GROUP,
        "counts": cube.layer("counts", FIXTURE_GROUP),
        "n_cells": cube.n_cells[:, 0],
    }
    for axis, table in (("sample", cube.samples), ("feature", cube.features)):
        arrays |= {f"{axis} {name}": table[name].astype(str) for name in table}
    arrays |= {f"limma {name}": np.asarray(values) for name, values in group.items()}
    arrays["limma aliased"] = arrays["limma aliased"].astype(str)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE / "limma.npz", **arrays)
    print(f"wrote limma.npz to {FIXTURE}")


def memory(restart=False):
    """This process's resident memory and its peak, in MiB, or NaN off Linux. With
    restart, the peak restarts from the resident memory first (Linux 4.0 or newer),
    and both are NaN if it cannot."""
    try:
        if restart:
            Path("/proc/self/clear_refs").write_text("5")
        status = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return np.nan, np.nan
    fields = dict(line.split(":", 1) for line in status if ":" in line)
    return tuple(int(fields[key].split()[0]) / 1024 for key in ("VmRSS", "VmHWM"))


def machine():
    """The CPU, logical CPUs, memory, operating system, numba threads and date."""
    cpu = platform.processor() or platform.machine()
    try:
        info = Path("/proc/cpuinfo").read_text().splitlines()
        cpu = next(line.split(":")[1].strip() for line in info if "model name" in line)
    except (OSError, StopIteration):
        pass
    gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return (
        f"{cpu}, {os.cpu_count()} logical CPUs, {gib:.0f} GiB; {platform.platform()}; "
        f"{numba.get_num_threads()} numba threads; {datetime.date.today()}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help="the directory of the GEO files")
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = Path(
        arguments.data
        or os.environ.get("GSE96583_DATA")
        or cache / "cytepool-validation" / "GSE96583"
    ).expanduser()
    if directory.resolve().is_relative_to(FIXTURE.parents[2]):
        sys.exit(f"{directory} is in the repository; keep the data outside it")
    clock, seconds = time.perf_counter(), {}
    fetch(directory)
    cells, labels, genes = read_cells(directory)
    seconds["reading and checking the files"] = time.perf_counter() - clock
    clock = time.perf_counter()
    cube = make_cube(cells, labels, genes)
    seconds["from_cells, pooling the cells"] = time.perf_counter() - clock
    verdict = {True: "pass", False: "FAIL"}
    differ = pooling_differences(cube, cells, labels)
    table = [
        ("from_cells: counts of every sample and cell type, as sums of cells", "scipy")
        + ("counts that differ", str(differ), "0", verdict[differ == 0])
    ]
    print(f"{cells.shape[0]:,} cells x {cells.shape[1]:,} genes, {cells.nnz:,} stored")
    print(f"counts, from {directory}; {cube}")
    del cells
    symbols = cube.features["symbol"]
    require({"XIST", *Y_GENES} <= set(symbols.tolist()), "XIST and Y genes are listed")
    totals = cube.layer("counts").sum(axis=0, dtype=np.int64)  # samples x features
    xist, y = (totals[:, np.isin(symbols, s)] for s in (["XIST"], Y_GENES))
    print(
        f"XIST is counted in {np.count_nonzero(xist.sum(axis=1))} of 16 samples; the "
        f"Y-chromosome genes {', '.join(Y_GENES)} have {y.sum()} counts"
    )

    loading, start = memory()[1], memory(restart=True)[0]
    restarted = not np.isnan(start)
    design = paired_design(cube)
    for call in ("first", "second"):
        clock = time.perf_counter()
        result = limma_voom(cube, design)
        seconds[f"limma_voom, 8 cell types, {call} call"] = time.perf_counter() - clock
    during = memory()[1] if restarted else np.nan
    print(f"limma_voom settings: {json.dumps(result.settings)}")
    cases = {"": (cube, design, result)}
    if arguments.write_fixtures:
        cases["fixture, "] = fixture_case(cube, result)
    clock = time.perf_counter()
    splits, calls = negative_control(cube, result)
    seconds["limma_voom, negative control"] = time.perf_counter() - clock
    limma, versions, r_peak = run_limma(cases)
    seconds["R, limma workflow, 8 cell types"] = limma[""][1]
    later = memory()[1] if restarted else np.nan

    print("Reference: " + ", ".join(versions))
    print(
        f"CytePool {cytepool.__version__}, Python {platform.python_version()}, numpy "
        f"{np.__version__}, numba {numba.__version__}, scipy {scipy.__version__}"
    )
    print(f"Machine: {machine()}")
    for label, (case_cube, _, case_result) in cases.items():
        table += compare(label, case_cube, case_result, limma[label][0])
    table += information(cube, result, splits, calls)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    print("Negative control, per cell type: the calls in each split; for each split")
    print("with calls, the donors of group A and the genes called, smallest p-value")
    print("first; the genes called in more than one split")
    for group, found in calls.items():
        print(f"- {group}: {' '.join(str(len(genes)) for genes in found)}")
        for (first, _), genes in zip(splits[group], found, strict=True):
            more = f" and {len(genes) - 8} more" if len(genes) > 8 else ""
            if genes:
                print(f"  {', '.join(first)}: {', '.join(genes[:8])}{more}")
        repeated = Counter(symbol for genes in found for symbol in genes)
        named = [f"{s} ({n})" for s, n in repeated.most_common() if n > 1]
        print(f"  called in more than one split: {', '.join(named) or 'none'}")
    print("| step | seconds |\n| --- | --- |")
    for step, value in seconds.items():
        print(f"| {step} | {value:.2f} |")
    print(
        f"Peak resident memory, MiB: this process {loading:,.0f} while reading and "
        f"pooling the cells; {during:,.0f} while limma_voom ran (from {start:,.0f} at "
        f"its start) and {later:,.0f} from then to the end; R {r_peak:,.0f}"
    )
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        write_fixtures(cases["fixture, "], limma["fixture, "][0])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
