"""Check check_dose_design against scanpy, R's qr(), limma's nonEstimable, ETAP's
minimums coded in R, and a published dose series with one participant per dose.

Counts (seeds 20261127 and 20261128): a TCDD-like series of 24 mice, the vehicle (4)
and 0.01 to 30 ug/kg (2 or 3 each), each mouse in a lane of two, with the cells of 4
cell types and 300 genes: hepatocytes, abundant; stellate cells, depleted with dose;
cholangiocytes, rare, about 9 cells per mouse; and macrophages, absent from the
vehicle mice. ``from_cells`` pools them, and ``check_dose_design`` reports, at
``min_counts`` 0 and at the median count of the stellate profiles of 10 cells or more,
each sample's cells and counts summed over all features and whether it is used, and
each dose group's samples used and left out and its fewest cells and counts. scanpy's
``get.aggregate`` sums the same cells per mouse and cell type (``sum``,
``n_obs_aggregated``), and pandas applies the filter (``min_cells`` 10, some counts,
``min_counts``) and groups the samples by cell type and dose. Tier: identical.

Aliasing and statuses: 25 designs, among them the TCDD-like series with its lanes; the
lane equal to the dose; two lanes that split the low and high doses; the vehicle in
both lanes; a lane per sample; eight lanes, two vehicle samples sharing one; two
dose-1 samples sharing a lane; the vehicle alone in its lane; one sample alone in its
lane; sex and lane that absorb dose 1 together under ``~ sex``; sex equal to the top
dose's indicator; one sample per dose; a replicated vehicle with single doses; a
series with one participant per dose sampled before and after dosing, under
``~ patient`` and ``~ patient + timepoint``; a bulk cube; the vehicle and one dose;
two covariates that leave no residual degrees of freedom; ``~ 0``; a covariate equal
to the treated indicator; no vehicle; and four designs with a random intercept: the
vehicle alone in its lane under ``(1 | lane)``, two lanes of their own per dose under a
random intercept that groups the samples as the lanes do, each treated dose in a lane
of its own under ``(1 | lane)``, and one animal of two samples per treated dose under
``(1 | animal)``. R filters each cell type's samples from their cells and counts,
builds the design's columns with ``model.matrix`` on every sample and keeps the rows
used, adds an intercept and an indicator of each treated dose group that keeps a
sample, and computes the rank with ``qr()``; a dose group is estimable when limma's
``nonEstimable`` does not name its column placed last, and confounded with the batch
when it is estimable and ``nonEstimable`` names it once contrasts of the batches join
the model, those of a single sample pooled into one and a sample alone in its batch,
when it is the only one, left at 0. The statuses follow the rules of
``cytepool.dose_designs`` coded again in R for this check: replicates counted as the
levels of a random intercept's grouping when the design has one, and a batch that a
random intercept models excused when the vehicle and each confounded dose span two
batches or more. The second coding catches slips in either one. CytePool runs on
cubes of one feature that hold the same cells and counts. Tier: identical.

ETAP's minimums (US EPA 2024, Sec. 3.4.1.3, 3.4.2 and 3.4.5.2), coded in R from the
text for this check (self-coded): the TCDD-like series, males only; a design of the
vehicle and 0.03 to 300 mg/kg with 4 males and 4 females per dose, with cell types
that keep every sample, keep one passing male at one dose, or keep 2 vehicle samples
of one sex; the same with 3 females at one dose; and 10 ladders whose steps are
changed to 2.5, 3, 10/3, 3.5, 3.6 and 4 at one step, and the lowest step to 3, 8.5, 9
and 10 (none within 1e-3 log10 of a bound). Each minimum's met and observed text must
be identical.

Published data: GEO series GSE208161 (Sonigra et al. 2022, JCI Insight 7, e160964),
the cells of a phase 1 trial with one participant per dose level, placebo included,
each sampled at day 1 (baseline) and day 29, as 18 Seurat clusters in the deposited
per-cell metadata. ``from_cells`` pools each cell's RNA count as the cube's one
feature, since the check reads cells and counts only, and the doses come from GEO's
sample records (GSM6338295 placebo, GSM6338292 0.3 mL, GSM6338289 1 mL, GSM6338298 3
mL). The cells and RNA counts of each sample and cluster must equal sums of the
metadata's rows, and with the baselines at dose 0 and ``~ patient``, every cluster
must be descriptive, since no dose has a second participant; information rows give
the statuses under ``~ 1`` and with the day-29 samples alone.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree, and access to ftp.ncbi.nlm.nih.gov
    for the first run (1.7 MB). scanpy with scipy, pandas and anndata in
    SCANPY_PYTHON, or in this interpreter when that is not set. R 4.4 or newer with
    limma and jsonlite; set RSCRIPT to the Rscript executable when it is not on PATH.
    The data directory is --data, else GSE208161_DATA, else
    cytepool-validation/GSE208161 under XDG_CACHE_HOME or ~/.cache; it must be
    outside the repository.

Usage:
    PYTHONPATH=. python validation/dose_design.py
    PYTHONPATH=. python validation/dose_design.py --write-fixtures
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from cytepool import Cube, Design, DoseAxis, check_dose_design, from_cells

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_design"
)
SEED = 20261127
LADDER = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
REPLICATES = (4, 2, 3, 3, 2, 3, 2, 3, 2)  # the vehicle and each dose
CELL_TYPES = ("hepatocyte", "stellate", "cholangiocyte", "macrophage")
GENES = 300
MIN_CELLS = 10
UG_PER_KG = DoseAxis("dose", unit="ug/kg", metric="administered")
MG_PER_KG = DoseAxis("dose", unit="mg/kg", metric="administered")
ML = DoseAxis("dose", unit="mL", metric="administered")
SOURCE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE208nnn/GSE208161/suppl/"
METADATA = "GSE208161_Pt04_Pt11_Pt19_Pt21_integrated_metadata.csv.gz"
# Bytes and SHA-256, as downloaded on 2026-11-20.
FILES = {
    METADATA: (
        1_679_182,
        "dd36521e006f18416d9c5a0db5741dd374966128bcd9b98510fd0a9c585af4fe",
    )
}
# Each participant's dose of DEN-181 at day 29, from GEO's sample records.
PARTICIPANT_DOSES = {
    "patient_19": 0.0,
    "patient_11": 0.3,
    "patient_04": 1.0,
    "patient_21": 3.0,
}

SCANPY_PROGRAM = textwrap.dedent(
    '''
    """Reference: sums and cells of every mouse and cell type by scanpy's aggregate;
    the filter of samples and their grouping by cell type and dose by pandas."""

    import json
    import sys
    import warnings
    from importlib.metadata import version

    import anndata
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    data = np.load(f"{directory}/cells.npz")
    settings = json.load(open(f"{directory}/settings.json"))
    shape = tuple(int(n) for n in data["shape"])
    matrix = sp.csr_matrix(
        (data["data"].astype(np.float64), data["indices"], data["indptr"]), shape=shape
    )
    obs = pd.DataFrame(
        {
            "mouse": pd.Categorical(data["mouse"]),
            "cell_type": pd.Categorical(data["cell_type"]),
        },
        index=[f"cell{i}" for i in range(shape[0])],
    )
    adata = anndata.AnnData(X=matrix, obs=obs)
    sums = sc.get.aggregate(adata, by=["mouse", "cell_type"], func="sum")
    found = pd.DataFrame(
        {
            "mouse": sums.obs["mouse"].astype(str).to_numpy(),
            "cell_type": sums.obs["cell_type"].astype(str).to_numpy(),
            "n_cells": sums.obs["n_obs_aggregated"].to_numpy(),
            "total_counts": np.asarray(sums.layers["sum"]).sum(axis=1),
        }
    )
    # Every mouse and cell type, with no cells where scanpy found none.
    grid = pd.MultiIndex.from_product(
        [settings["mice"], settings["cell_types"]], names=["mouse", "cell_type"]
    )
    table = (
        found.set_index(["mouse", "cell_type"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )
    table["dose"] = table["mouse"].map(dict(zip(settings["mice"], settings["doses"])))
    sum_rows = {key: row for row, key in enumerate(zip(found["mouse"], found["cell_type"]))}
    layer = np.asarray(sums.layers["sum"])
    grid_sums = np.zeros((len(settings["cell_types"]), len(settings["mice"]), shape[1]))
    for g, cell_type in enumerate(settings["cell_types"]):
        for i, mouse in enumerate(settings["mice"]):
            if (mouse, cell_type) in sum_rows:
                grid_sums[g, i] = layer[sum_rows[mouse, cell_type]]
    results = []
    for min_counts in settings["min_counts"]:
        used = (
            (table["n_cells"] >= settings["min_cells"])
            & (table["total_counts"] > 0)
            & (table["total_counts"] >= min_counts)
        )
        frame = table.assign(used=used)
        groups = []
        for (cell_type, dose), group in frame.groupby(["cell_type", "dose"], sort=False):
            kept = group[group["used"]]
            groups.append(
                {
                    "cell_type": cell_type,
                    "dose": float(dose),
                    "n_samples_used": int(len(kept)),
                    "n_samples_left_out": int(len(group) - len(kept)),
                    "n_cells_min": float(kept["n_cells"].min()) if len(kept) else None,
                    "total_counts_min": (
                        float(kept["total_counts"].min()) if len(kept) else None
                    ),
                }
            )
        results.append(
            {
                "min_counts": min_counts,
                "samples": {
                    "cell_type": frame["cell_type"].tolist(),
                    "mouse": frame["mouse"].tolist(),
                    "n_cells": frame["n_cells"].astype(int).tolist(),
                    "total_counts": frame["total_counts"].astype(float).tolist(),
                    "used": frame["used"].astype(bool).tolist(),
                },
                "groups": groups,
            }
        )
    np.save(f"{directory}/sums.npy", grid_sums)
    json.dump(
        {
            "results": results,
            "versions": {
                name: version(name) for name in ("scanpy", "pandas", "anndata")
            },
        },
        open(f"{directory}/scanpy.json", "w"),
    )
    '''
)

R_SCRIPT = r"""
suppressPackageStartupMessages({ library(limma); library(jsonlite) })
directory <- commandArgs(trailingOnly = TRUE)[[1]]
input <- fromJSON(file.path(directory, "input.json"), simplifyVector = TRUE,
                  simplifyDataFrame = FALSE)
used_samples <- function(cell_type, min_cells, min_counts) {
  totals <- as.numeric(cell_type$totals)
  used <- totals > 0 & totals >= min_counts
  if (!is.null(cell_type$cells)) used <- used & as.numeric(cell_type$cells) >= min_cells
  used
}
# R's qr() places aliased columns last and limma's nonEstimable names them: the
# column placed last is estimable when it is not named.
aliased_last <- function(base, columns, k) {
  z <- cbind(base, columns[, -k, drop = FALSE], last = columns[, k])
  "last" %in% nonEstimable(z)
}
check <- function(case, cell_type) {
  samples <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  dose <- as.numeric(samples$dose)
  levels <- sort(unique(dose))
  used <- used_samples(cell_type, case$min_cells, case$min_counts)
  out <- list(used = used, df_residual = NA, estimable = rep(FALSE, length(levels)),
              confounded = rep(FALSE, length(levels)), status = "refused")
  if (!any(used)) return(out)
  C <- model.matrix(as.formula(case$formula), samples)[used, , drop = FALSE]
  colnames(C) <- sprintf("c%d", seq_len(ncol(C)))
  d <- dose[used]
  kept <- levels[levels > 0 & levels %in% d]
  D <- matrix(0, sum(used), length(kept), dimnames = list(NULL, sprintf("d%d", seq_along(kept))))
  for (k in seq_along(kept)) D[, k] <- as.numeric(d == kept[k])
  base <- cbind(C, one = 1)
  out$df_residual <- sum(used) - qr(cbind(base, D))$rank
  if (levels[1] != 0 || !(0 %in% d) || length(kept) == 0) return(out)
  estimable <- !vapply(seq_along(kept), function(k) aliased_last(base, D, k), logical(1))
  out$estimable[match(kept, levels)] <- estimable
  if (!any(estimable)) return(out)
  compared <- TRUE
  if (!is.null(case$batch)) {
    labels <- as.character(samples[[case$batch]][used])
    sizes <- as.vector(table(labels)[labels])
    pooled <- ifelse(sizes > 1, labels, "(a single sample)")
    batches <- unique(pooled)
    # A sample alone in its batch, the only one, joins no batch: contrasts that sum
    # to zero over the other batches leave it at 0.
    if (sum(table(labels) == 1) == 1) batches <- setdiff(batches, "(a single sample)")
    if (length(batches) > 1) {
      last <- as.numeric(pooled == batches[length(batches)])
      B <- vapply(batches[-length(batches)], function(b) as.numeric(pooled == b) - last,
                  numeric(length(pooled)))
      B <- matrix(B, nrow = length(pooled), dimnames = list(NULL, sprintf("b%d", seq_len(length(batches) - 1))))
      confounded <- estimable & vapply(seq_along(kept),
        function(k) aliased_last(cbind(base, B), D, k), logical(1))
      out$confounded[match(kept, levels)] <- confounded
      # A random intercept that groups the samples as the batch does compares the
      # confounded doses with the vehicle when the vehicle and each span two batches.
      all_labels <- as.character(samples[[case$batch]])
      modelled <- any(vapply(case$groupings, function(g) {
        grouping <- as.character(samples[[g]])
        pairs <- length(unique(paste(all_labels, grouping, sep = "\r")))
        pairs == length(unique(all_labels)) && pairs == length(unique(grouping))
      }, logical(1)))
      spans <- vapply(c(0, kept), function(t) length(unique(labels[d == t])), integer(1))
      compared <- !any(confounded) ||
        (modelled && spans[1] >= 2 && all(spans[-1][confounded] >= 2))
    }
  }
  # Replicates: samples, or the fewest levels of a random intercept's grouping.
  replicates <- vapply(kept, function(t) sum(d == t), integer(1))
  for (g in case$groupings) {
    units <- as.character(samples[[g]][used])
    replicates <- pmin(replicates, vapply(kept, function(t) length(unique(units[d == t])), integer(1)))
  }
  out$status <- if (!any(replicates[estimable] >= 2)) "descriptive" else
    if (out$df_residual == 0) "descriptive" else
    if (!compared) "descriptive" else "inferential"
  out
}
count_text <- function(n, noun) paste0(n, " ", noun, if (n == 1) "" else "s")
shortfalls <- function(doses, counts, required, where, unit) {
  short <- counts < required
  if (any(short)) {
    paste(paste0(counts[short], " at ", sprintf("%g", doses[short]), " ", unit), collapse = ", ")
  } else if (length(counts) == 0) "no doses" else paste0("at least ", min(counts), " ", where)
}
# US EPA (2024) Sec. 3.4.1.3, 3.4.2 and 3.4.5.2, coded from the text.
etap <- function(design) {
  dose <- as.numeric(design$samples$dose)
  sex <- as.character(design$samples$sex)
  unit <- design$unit
  rows <- list()
  add <- function(group, s, minimum, observed, met) {
    rows[[length(rows) + 1]] <<- list(group_id = group, sex = s, minimum = minimum,
                                      observed = observed, met = met)
  }
  levels <- sort(unique(dose))
  positive <- levels[levels > 0]
  vehicle <- 0 %in% levels
  add(NA, NA, "dose_levels",
      paste0(count_text(length(positive), "dose"), " and ", if (vehicle) "a" else "no", " vehicle"),
      length(positive) >= 8 && vehicle)
  if (length(positive) < 2) {
    add(NA, NA, "dose_spacing", "fewer than two doses", FALSE)
  } else {
    steps <- diff(log10(positive))
    add(NA, NA, "dose_spacing",
        paste0("steps of ", paste(sprintf("%.2f", rev(steps)), collapse = ", "),
               " log10 down from the highest dose"),
        abs(steps[1] - 1) <= 0.05 && all(abs(steps[-1] - 0.5) <= 0.05))
  }
  sexes <- sort(unique(sex))
  add(NA, NA, "sexes", paste0(length(sexes), " (", paste(sexes, collapse = ", "), ")"),
      length(sexes) >= 2)
  for (s in sexes) {
    counts <- vapply(levels, function(l) sum(dose == l & sex == s), integer(1))
    add(NA, s, "animals_per_sex_and_dose",
        shortfalls(levels, counts, 4, "in each dose group", unit), min(counts) >= 4)
  }
  for (cell_type in design$cell_types) {
    used <- used_samples(cell_type, design$min_cells, design$min_counts)
    for (s in sexes) {
      counts <- vapply(positive, function(l) sum(used & dose == l & sex == s), integer(1))
      add(cell_type$name, s, "passing_samples_per_dose",
          shortfalls(positive, counts, 2, "at each dose", unit),
          length(counts) == 0 || min(counts) >= 2)
      vehicle_samples <- sum(used & dose == 0 & sex == s)
      add(cell_type$name, s, "passing_vehicle_samples", as.character(vehicle_samples),
          vehicle_samples >= 3)
    }
  }
  rows
}
aliasing <- lapply(input$cases, function(case) {
  lapply(case$cell_types, function(cell_type) check(case, cell_type))
})
minimums <- lapply(input$designs, etap)
version <- function(name) paste(name, packageDescription(name, fields = "Version"))
writeLines(toJSON(list(aliasing = aliasing, minimums = minimums,
                       versions = paste0("R ", getRversion(), " (qr), ", version("limma"),
                                         " (nonEstimable)")),
                  auto_unbox = TRUE, na = "null", digits = NA, null = "null"),
           file.path(directory, "r.json"))
"""


# The TCDD-like series.


def tcdd_series() -> dict:
    """The mice of the series: ids, doses, lanes of two mice and sexes."""
    rng = np.random.default_rng(SEED)
    doses = np.repeat(np.r_[0.0, LADDER], REPLICATES)
    lanes = [f"L{position // 2 + 1}" for position in rng.permutation(doses.size)]
    return {
        "sample_id": [f"M{i + 1:02d}" for i in range(doses.size)],
        "dose": doses,
        "lane": lanes,
        "sex": ["M"] * doses.size,
    }


def simulate_cells(series: dict) -> dict:
    """Cells x genes counts of every mouse and cell type, with their labels."""
    rng = np.random.default_rng(SEED + 1)
    doses = series["dose"]
    step = np.searchsorted(np.r_[0.0, LADDER], doses)  # 0 for the vehicle
    means = np.column_stack(
        [
            np.full(doses.size, 250.0),
            60.0 * 0.75**step,
            np.full(doses.size, 9.0),
            np.where(doses > 0.0, 4.0 * step, 0.0),
        ]
    )
    cells = rng.poisson(means)
    mouse = np.repeat(np.repeat(series["sample_id"], len(CELL_TYPES)), cells.ravel())
    cell_type = np.repeat(np.tile(CELL_TYPES, doses.size), cells.ravel())
    gene_means = rng.gamma(0.6, 2.0, GENES)
    size = rng.lognormal(0.0, 0.3, mouse.size)
    counts = rng.poisson(np.outer(size, gene_means))
    order = rng.permutation(mouse.size)
    return {
        "counts": counts[order],
        "mouse": mouse[order].astype(str),
        "cell_type": cell_type[order].astype(str),
    }


def cells_and_totals(cells: dict, series: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cells and counts over every gene of each mouse and cell type, by numpy."""
    mice = {mouse: i for i, mouse in enumerate(series["sample_id"])}
    types = {name: g for g, name in enumerate(CELL_TYPES)}
    rows = np.array([mice[m] for m in cells["mouse"]])
    columns = np.array([types[t] for t in cells["cell_type"]])
    n_cells = np.zeros((len(mice), len(types)), dtype=np.int64)
    totals = np.zeros((len(mice), len(types)), dtype=np.int64)
    np.add.at(n_cells, (rows, columns), 1)
    np.add.at(totals, (rows, columns), cells["counts"].sum(axis=1))
    return n_cells, totals


def counts_cube(cells: dict, series: dict) -> Cube:
    return from_cells(
        cells["counts"],
        sample_labels=cells["mouse"],
        group_labels=cells["cell_type"],
        samples=series,
        groups={"group_id": list(CELL_TYPES)},
        layers="counts",
    )


def stellate_threshold(n_cells: np.ndarray, totals: np.ndarray) -> float:
    """The median count of the stellate profiles of 10 cells or more."""
    stellate = CELL_TYPES.index("stellate")
    kept = n_cells[:, stellate] >= MIN_CELLS
    return float(np.round(np.median(totals[kept, stellate])))


def run_scanpy(cells: dict, series: dict, min_counts: list[float]) -> tuple[dict, str]:
    """scanpy's sums and pandas' tables, and the versions."""
    python = os.environ.get("SCANPY_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        counts = cells["counts"]
        rows, columns = np.nonzero(counts)
        indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=counts.shape[0]))]
        np.savez(
            directory / "cells.npz",
            data=counts[rows, columns],
            indices=columns,
            indptr=indptr,
            shape=np.array(counts.shape),
            mouse=cells["mouse"],
            cell_type=cells["cell_type"],
        )
        settings = {
            "mice": list(series["sample_id"]),
            "doses": series["dose"].tolist(),
            "cell_types": list(CELL_TYPES),
            "min_cells": MIN_CELLS,
            "min_counts": min_counts,
        }
        (directory / "settings.json").write_text(json.dumps(settings))
        (directory / "reference.py").write_text(SCANPY_PROGRAM)
        subprocess.run(
            [python, "-I", str(directory / "reference.py"), str(directory)], check=True
        )
        found = json.loads((directory / "scanpy.json").read_text())
        found["sums"] = np.load(directory / "sums.npy")
    versions = ", ".join(f"{name} {v}" for name, v in found["versions"].items())
    return found, versions


def compare_counts(cube: Cube, series: dict, reference: dict) -> list[tuple]:
    """CytePool's samples and doses tables against pandas', per setting."""
    rows = []
    design = Design(cube.samples, "~ 1")
    for result in reference["results"]:
        report = check_dose_design(
            cube, design, dose=UG_PER_KG, min_counts=result["min_counts"]
        )
        ours = report.samples
        theirs = result["samples"]
        index = {
            (t, m): i
            for i, (t, m) in enumerate(zip(theirs["cell_type"], theirs["mouse"]))
        }
        order = [
            index[t, m]
            for t, m in zip(ours["group_id"].tolist(), ours["sample_id"].tolist())
        ]
        mismatches = 0
        for column in ("n_cells", "total_counts", "used"):
            expected = np.asarray(theirs[column])[order]
            mismatches += int(np.sum(ours[column] != expected))
        setting = f"min_counts {result['min_counts']:g}"
        rows.append(
            (
                f"Cells, counts and use of each sample in each cell type, {setting} "
                f"({len(order)} samples)",
                "scanpy aggregate, pandas",
                f"{mismatches} mismatches",
                "0",
                "pass" if mismatches == 0 else "FAIL",
            )
        )
        doses = report.doses
        groups = {(g["cell_type"], g["dose"]): g for g in result["groups"]}
        mismatches = 0
        for row in range(len(doses)):
            group = groups[doses["group_id"][row], float(doses["dose"][row])]
            for column in (
                "n_samples_used",
                "n_samples_left_out",
                "n_cells_min",
                "total_counts_min",
            ):
                expected = group[column]
                value = doses[column][row]
                if expected is None:
                    mismatches += int(not np.isnan(value))
                else:
                    mismatches += int(value != expected)
        rows.append(
            (
                f"Samples used and left out, fewest cells and counts of each dose group "
                f"in each cell type, {setting} ({len(doses)} groups)",
                "scanpy aggregate, pandas",
                f"{mismatches} mismatches",
                "0",
                "pass" if mismatches == 0 else "FAIL",
            )
        )
    return rows


# Designs for the aliasing and the statuses.


def simple(
    name: str,
    doses: list[float],
    formula: str = "~ 1",
    batch: str | None = None,
    cell_types: dict | None = None,
    min_counts: float = 0.0,
    grouping: str | None = None,
    **columns: list,
) -> dict:
    """A design of a few samples: every sample with 30 cells and 1,000 counts or more
    in the cell type "kept", unless ``cell_types`` gives cells and counts; with
    ``grouping``, a random intercept of that column."""
    n = len(doses)
    samples = {"sample_id": [f"S{i + 1}" for i in range(n)], "dose": doses, **columns}
    if cell_types is None:
        cell_types = {"kept": ([30] * n, [1000 + 10 * i for i in range(n)])}
    return {
        "name": name,
        "samples": samples,
        "formula": formula,
        "random": None if grouping is None else f"(1 | {grouping})",
        "groupings": [] if grouping is None else [grouping],
        "batch": batch,
        "min_cells": MIN_CELLS,
        "min_counts": min_counts,
        "cell_types": [
            {"name": key, "cells": cells, "totals": totals}
            for key, (cells, totals) in cell_types.items()
        ],
    }


def aliasing_cases(series: dict, n_cells: np.ndarray, totals: np.ndarray) -> list[dict]:
    dose9 = [0, 0, 0, 1, 1, 10, 10, 30, 30]
    # The dose-1 samples fall below min_cells in "filtered".
    filtered = {
        "kept": ([30] * 9, [1000 + 10 * i for i in range(9)]),
        "filtered": (
            [30, 30, 30, 4, 30, 30, 30, 30, 30],
            [1000 + 10 * i for i in range(9)],
        ),
    }
    participants = ["P19", "P11", "P04", "P21"] * 2
    tcdd = {
        "name": "TCDD-like series, lanes of two mice",
        "samples": {
            "sample_id": series["sample_id"],
            "dose": series["dose"].tolist(),
            "lane": series["lane"],
        },
        "formula": "~ 1",
        "random": None,
        "groupings": [],
        "batch": "lane",
        "min_cells": MIN_CELLS,
        "min_counts": 0.0,
        "cell_types": [
            {
                "name": name,
                "cells": n_cells[:, g].tolist(),
                "totals": totals[:, g].tolist(),
            }
            for g, name in enumerate(CELL_TYPES)
        ],
    }
    return [
        tcdd,
        simple(
            "lane equal to the dose",
            dose9,
            batch="lane",
            cell_types=filtered,
            lane=list("aaabbccdd"),
        ),
        simple(
            "two lanes, low and high doses",
            dose9,
            batch="lane",
            cell_types=filtered,
            lane=list("aaaaabbbb"),
        ),
        simple(
            "the vehicle in both lanes",
            dose9,
            batch="lane",
            cell_types=filtered,
            lane=list("abaababba"),
        ),
        simple(
            "a lane per sample", dose9, batch="lane", lane=[f"L{i}" for i in range(9)]
        ),
        simple(
            "eight lanes, two vehicle samples sharing one",
            dose9,
            batch="lane",
            lane=["L0", "L0", *[f"L{i}" for i in range(2, 9)]],
        ),
        simple(
            "two dose-1 samples sharing a lane",
            dose9,
            batch="lane",
            lane=["L0", "L1", "L2", "L3", "L3", *[f"L{i}" for i in range(5, 9)]],
        ),
        simple(
            "the vehicle alone in its lane",
            dose9,
            batch="lane",
            cell_types=filtered,
            lane=list("aaabcbcbc"),
        ),
        simple(
            "one sample alone in its lane",
            [0, 0, 0, 0, 1, 1, 10, 10, 30],
            batch="lane",
            lane=list("abcdabcde"),
        ),
        simple(
            "the vehicle alone in its lane, (1 | lane)",
            [0, 0, 0, 1, 1, 1, 10, 10, 10],
            batch="lane",
            grouping="lane",
            lane=list("VVVABABAB"),
        ),
        simple(
            "two lanes of their own per dose, (1 | lane_id) grouping as the lanes",
            [0, 0, 0, 0, 1, 1, 1, 1, 10, 10, 10, 10],
            batch="lane",
            grouping="lane_id",
            lane=list("aabbccddeeff"),
            lane_id=[f"lane {label}" for label in "aabbccddeeff"],
        ),
        simple(
            "each treated dose in a lane of its own, (1 | lane)",
            [0, 0, 0, 0, 1, 1, 10, 10],
            batch="lane",
            grouping="lane",
            lane=list("aabbccdd"),
        ),
        simple(
            "one animal of two samples per treated dose, (1 | animal)",
            [0, 0, 0, 0, 1, 1, 10, 10, 30, 30],
            grouping="animal",
            animal=["A1", "A1", "A2", "A2", "A3", "A3", "A4", "A4", "A5", "A5"],
        ),
        simple(
            "~ sex: sex and lane absorb dose 1 together",
            [0, 0, 1, 1, 1, 1, 10, 10],
            formula="~ sex",
            batch="lane",
            sex=["F", "F", "M", "M", "F", "F", "F", "F"],
            lane=list("aaaabbaa"),
        ),
        simple(
            "~ sex: sex equal to the top dose's indicator",
            [0, 0, 1, 1, 10, 10, 30, 30],
            formula="~ sex",
            sex=["F"] * 6 + ["M"] * 2,
        ),
        simple("one sample per dose", [0, 1, 10, 30]),
        simple("a replicated vehicle and single doses", [0, 0, 0, 0, 1, 10, 30]),
        simple(
            "before and after dosing, one participant per dose, ~ patient",
            [0, 0, 0, 0, 0, 0.3, 1, 3],
            formula="~ patient",
            patient=participants,
        ),
        simple(
            "before and after dosing, ~ patient + timepoint",
            [0, 0, 0, 0, 0, 0.3, 1, 3],
            formula="~ patient + timepoint",
            patient=participants,
            timepoint=["day1"] * 4 + ["day29"] * 4,
        ),
        simple(
            "a bulk cube, min_counts 1000",
            dose9,
            cell_types={
                "bulk": (None, [1200, 0, 1100, 900, 1000, 950, 1020, 980, 1010])
            },
            min_counts=1000.0,
        ),
        simple("the vehicle and one dose", [0, 0, 1, 1]),
        simple(
            "two covariates that leave no residual df",
            [0, 0, 1, 1],
            formula="~ a + b",
            a=[0.1, 0.5, 0.2, 0.9],
            b=[1.0, 0.3, 0.7, 0.2],
        ),
        simple("~ 0", [0, 0, 0, 1, 1, 10, 10], formula="~ 0"),
        simple(
            "a covariate equal to the treated indicator",
            [0, 0, 1, 1, 10, 10],
            formula="~ x",
            x=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ),
        simple("no vehicle", [1, 1, 10, 10]),
    ]


def case_cube(case: dict, cells_by_type: list | None = None) -> Cube:
    """A cube of one feature that holds each sample's cells and counts."""
    totals = np.array([c["totals"] for c in case["cell_types"]], dtype=np.int64)
    cells = [c["cells"] for c in case["cell_types"]]
    n_cells = None
    if cells[0] is not None:
        n_cells = np.array(cells, dtype=np.int64).T
    return Cube(
        {"counts": totals[:, :, None]},
        samples=case["samples"],
        groups={"group_id": [c["name"] for c in case["cell_types"]]},
        features={"feature_id": ["all"]},
        n_cells=n_cells,
    )


def compare_aliasing(cases: list[dict], results: list, tcdd: Cube) -> list[tuple]:
    rows = []
    for case, found in zip(cases, results, strict=True):
        cube = tcdd if case["name"].startswith("TCDD") else case_cube(case)
        design = Design(case["samples"], case["formula"], random=case["random"])
        report = check_dose_design(
            cube,
            design,
            dose=UG_PER_KG,
            batch=case["batch"],
            min_cells=case["min_cells"],
            min_counts=case["min_counts"],
        )
        mismatches = []
        for g, (cell_type, reference) in enumerate(
            zip(case["cell_types"], found, strict=True)
        ):
            name = cell_type["name"]
            used = report.samples.filter(group_id=name)["used"]
            doses = report.doses.filter(group_id=name)
            summary = report.summary.filter(group_id=name)
            checks = {
                "used": (used, reference["used"]),
                "df_residual": (summary["df_residual"], [reference["df_residual"]]),
                "estimable": (doses["estimable"], reference["estimable"]),
                "status": (summary["status"], [reference["status"]]),
            }
            if case["batch"] is not None:
                checks["confounded"] = (
                    doses["confounded_with_batch"],
                    reference["confounded"],
                )
            for label, (ours, theirs) in checks.items():
                theirs = np.array([np.nan if v is None else v for v in theirs])
                same = (
                    np.array_equal(ours, theirs, equal_nan=True)
                    if label == "df_residual"
                    else np.array_equal(ours, theirs)
                )
                if not same:
                    mismatches.append(
                        f"{name} {label}: {ours.tolist()} vs {theirs.tolist()}"
                    )
        statuses = ", ".join(
            f"{s} {n}" for s, n in Counter(report.summary["status"].tolist()).items()
        )
        rows.append(
            (
                f"{case['name']}: samples used, residual df, estimable dose groups"
                + (", groups confounded with the batch" if case["batch"] else "")
                + f", statuses ({statuses})",
                "qr, nonEstimable; statuses self-coded",
                f"{len(mismatches)} mismatches",
                "0",
                "pass" if not mismatches else "FAIL",
            )
        )
        for line in mismatches:
            print(f"differs: {case['name']}: {line}")
    return rows


# ETAP's minimums.


def etap_ladder(lowest_step: float = 10.0, step_factor: float | None = None) -> list:
    """The vehicle and 8 doses down from 300 by half a log10, the lowest a full log10
    below the second lowest; ``step_factor`` replaces the fourth step from the top."""
    doses = [300.0]
    for k in range(6):
        factor = step_factor if (step_factor is not None and k == 3) else 10**0.5
        doses.append(doses[-1] / factor)
    doses.append(doses[-1] / lowest_step)
    return sorted(doses)


def etap_designs(series: dict, n_cells: np.ndarray, totals: np.ndarray) -> list[dict]:
    def balanced(doses: list, per_sex: int, short: tuple | None = None) -> dict:
        dose, sex = [], []
        for level in [0.0, *doses]:
            for s in ("F", "M"):
                count = per_sex
                if short is not None and (level, s) == short:
                    count -= 1
                dose += [level] * count
                sex += [s] * count
        n = len(dose)
        return {
            "sample_id": [f"R{i + 1:02d}" for i in range(n)],
            "dose": dose,
            "sex": sex,
        }

    def design(name: str, samples: dict, cell_types: dict, unit: str = "mg/kg") -> dict:
        return {
            "name": name,
            "samples": samples,
            "unit": unit,
            "min_cells": MIN_CELLS,
            "min_counts": 0.0,
            "cell_types": [
                {"name": key, "cells": cells, "totals": counts}
                for key, (cells, counts) in cell_types.items()
            ],
        }

    ladder = [0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    full = balanced(ladder, 4)
    n = len(full["dose"])
    # Three of the four males at 1 mg/kg fail the filter, which leaves one passing
    # sample there; and two of the four female vehicle samples.
    one_left = [30] * n
    males = [
        i
        for i, (d, s) in enumerate(zip(full["dose"], full["sex"]))
        if d == 1 and s == "M"
    ]
    for i in males[:3]:
        one_left[i] = 3
    vehicle_female = [
        i
        for i, (d, s) in enumerate(zip(full["dose"], full["sex"]))
        if d == 0 and s == "F"
    ]
    two_vehicle = [30] * n
    for i in vehicle_female[:2]:
        two_vehicle[i] = 3
    counts = [1000] * n
    designs = [
        design(
            "TCDD-like series, males only",
            {
                "sample_id": series["sample_id"],
                "dose": series["dose"].tolist(),
                "sex": series["sex"],
            },
            {
                name: (n_cells[:, g].tolist(), totals[:, g].tolist())
                for g, name in enumerate(CELL_TYPES)
            },
            unit="ug/kg",
        ),
        design(
            "ETAP design: 4 males and 4 females per dose, 0.03 to 300 mg/kg",
            full,
            {
                "every sample": ([30] * n, counts),
                "one passing male at 1 mg/kg": (one_left, counts),
                "two female vehicle samples": (two_vehicle, counts),
            },
        ),
        design(
            "ETAP design with 3 females at 3 mg/kg",
            balanced(ladder, 4, short=(3.0, "F")),
            {"every sample": ([30] * (n - 1), [1000] * (n - 1))},
        ),
    ]
    for factor in (2.5, 3.0, 10 / 3, 3.5, 3.6, 4.0):
        samples = balanced(etap_ladder(step_factor=factor), 1)
        designs.append(
            design(
                f"a step of {factor:.4g} in an ETAP ladder",
                samples,
                {"every sample": ([30] * 18, [1000] * 18)},
            )
        )
    for lowest in (3.0, 8.5, 9.0, 10.0):
        samples = balanced(etap_ladder(lowest_step=lowest), 1)
        designs.append(
            design(
                f"a lowest step of {lowest:g} in an ETAP ladder",
                samples,
                {"every sample": ([30] * 18, [1000] * 18)},
            )
        )
    return designs


def compare_minimums(designs: list[dict], results: list) -> list[tuple]:
    rows = []
    for design, found in zip(designs, results, strict=True):
        axis = DoseAxis("dose", unit=design["unit"], metric="administered")
        cube = case_cube(design)
        report = check_dose_design(
            cube,
            Design(design["samples"], "~ 1"),
            dose=axis,
            rules="etap_2024",
            sex="sex",
            min_cells=design["min_cells"],
        )
        table = report.minimums
        assert table is not None
        ours = {
            (table["group_id"][i], table["sex"][i], table["minimum"][i]): (
                bool(table["met"][i]),
                table["observed"][i],
            )
            for i in range(len(table))
        }
        theirs = {
            (row["group_id"], row["sex"], row["minimum"]): (row["met"], row["observed"])
            for row in found
        }
        mismatches = sorted(
            str(key)
            for key in set(ours) | set(theirs)
            if ours.get(key) != theirs.get(key)
        )
        unmet = sum(not met for met, _ in ours.values())
        rows.append(
            (
                f"{design['name']}: met and observed of every minimum ({len(ours)} rows, "
                f"{unmet} unmet)",
                "ETAP's minimums in R (self-coded)",
                f"{len(mismatches)} mismatches",
                "0",
                "pass" if not mismatches else "FAIL",
            )
        )
        for key in mismatches:
            print(
                f"differs: {design['name']}: {key}: {ours.get(key)} vs {theirs.get(key)}"
            )
    return rows


def run_r(cases: list[dict], designs: list[dict]) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "input.json").write_text(
            json.dumps({"cases": cases, "designs": designs})
        )
        (directory / "check.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(directory / "check.R"), str(directory)],
            check=True,
        )
        found = json.loads((directory / "r.json").read_text())
    return found, found["versions"]


# GSE208161.


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(directory: Path) -> Path:
    """Download the metadata unless the directory holds it, checking its SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA
    size, checksum = FILES[METADATA]
    if not path.exists():
        print(f"downloading {SOURCE}{METADATA} ({size:,} bytes) to {directory}")
        partial = directory / f"{METADATA}.part"
        with urllib.request.urlopen(SOURCE + METADATA, timeout=120) as response:
            with partial.open("wb") as file:
                shutil.copyfileobj(response, file, 1 << 20)
        if sha256(partial) != checksum:
            partial.unlink()
            sys.exit(f"the download of {METADATA} does not have SHA-256 {checksum}")
        partial.replace(path)
    found = sha256(path)
    if found != checksum:
        sys.exit(
            f"{path} has SHA-256 {found}, not {checksum}; delete it to download again"
        )
    return path


def read_metadata(path: Path) -> dict:
    """The sample, participant, day, cluster and RNA count of every cell."""
    with gzip.open(path, "rt", newline="") as text:
        rows = list(csv.DictReader(text))
    return {
        "sample": np.array([row["uniqueID"] for row in rows]),
        "patient": np.array([row["patient"] for row in rows]),
        "day": np.array([row["timepoint"] for row in rows]),
        "cluster": np.array([row["seurat_clusters"] for row in rows]),
        "rna": np.array(
            [int(float(row["nCount_RNA"])) for row in rows], dtype=np.int64
        ),
    }


def gse208161(metadata: dict) -> tuple[list[tuple], dict]:
    samples = sorted(set(metadata["sample"].tolist()))
    clusters = sorted(set(metadata["cluster"].tolist()), key=int)
    patient = {s: p for s, p in zip(metadata["sample"], metadata["patient"])}
    day = {s: d for s, d in zip(metadata["sample"], metadata["day"])}
    table = {
        "sample_id": samples,
        "patient": [patient[s] for s in samples],
        "timepoint": [day[s] for s in samples],
        "dose": [
            0.0 if day[s] == "day1" else PARTICIPANT_DOSES[patient[s]] for s in samples
        ],
    }
    cube = from_cells(
        metadata["rna"][:, None],
        sample_labels=metadata["sample"],
        group_labels=metadata["cluster"],
        samples=table,
        groups={"group_id": clusters},
        layers="counts",
    )
    # Cells and RNA counts of each sample and cluster, from the rows of the metadata.
    cells = Counter(zip(metadata["sample"].tolist(), metadata["cluster"].tolist()))
    rna: Counter = Counter()
    for s, c, value in zip(
        metadata["sample"].tolist(), metadata["cluster"].tolist(), metadata["rna"]
    ):
        rna[s, c] += int(value)
    n_cells = np.array([[cells[s, c] for c in clusters] for s in samples])
    totals = np.array([[rna[s, c] for c in clusters] for s in samples])
    by_patient = check_dose_design(cube, Design(cube.samples, "~ patient"), dose=ML)
    found = by_patient.samples
    order = [
        (samples.index(s), clusters.index(c))
        for s, c in zip(found["sample_id"].tolist(), found["group_id"].tolist())
    ]
    mismatches = int(
        np.sum(found["n_cells"] != np.array([n_cells[i, j] for i, j in order]))
        + np.sum(found["total_counts"] != np.array([totals[i, j] for i, j in order]))
    )
    statuses = Counter(by_patient.summary["status"].tolist())
    descriptive = statuses["descriptive"] == len(clusters)
    df = Counter(by_patient.summary["df_residual"].tolist())
    plain = Counter(
        check_dose_design(cube, Design(cube.samples, "~ 1"), dose=ML)
        .summary["status"]
        .tolist()
    )
    day29 = [s for s in samples if day[s] == "day29"]
    late = check_dose_design(
        cube,
        Design(
            {k: [v for v, s in zip(table[k], samples) if s in day29] for k in table},
            "~ 1",
        ),
        dose=ML,
    )
    late_statuses = Counter(late.summary["status"].tolist())
    refused = late.summary.filter(status="refused")["group_id"].tolist()

    def text(counts: Counter) -> str:
        return ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))

    rows = [
        (
            f"GSE208161: cells and RNA counts of {len(samples)} samples x "
            f"{len(clusters)} clusters",
            "sums of the metadata's rows",
            f"{mismatches} mismatches",
            "0",
            "pass" if mismatches == 0 else "FAIL",
        ),
        (
            "GSE208161, baselines at dose 0, ~ patient: every cluster descriptive "
            f"(residual df: {', '.join(f'{n} with {d:g}' for d, n in sorted(df.items()))})",
            "the published design: one participant per dose level, placebo included",
            text(statuses),
            f"{len(clusters)} descriptive",
            "pass" if descriptive else "FAIL",
        ),
        (
            "GSE208161, baselines at dose 0, ~ 1",
            "information",
            text(plain),
            "n/a",
            "info",
        ),
        (
            "GSE208161, day-29 samples alone, ~ 1"
            + (f" (refused: clusters {', '.join(refused)})" if refused else ""),
            "information",
            text(late_statuses),
            "n/a",
            "info",
        ),
    ]
    fixture = {
        "sample_id": np.array(samples),
        "cluster": np.array(clusters),
        "n_cells": n_cells,
        "rna": totals,
        "patient": np.array(table["patient"]),
        "timepoint": np.array(table["timepoint"]),
        "dose": np.array(table["dose"]),
    }
    return rows, fixture


def write_fixtures(
    series: dict,
    n_cells: np.ndarray,
    scanpy: dict,
    cases: list[dict],
    r: dict,
    designs: list[dict],
    gse: dict,
) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sums = scanpy["sums"]
    arrays = {
        "sample_id": np.array(series["sample_id"]),
        "dose": series["dose"],
        "lane": np.array(series["lane"]),
        "cell_type": np.array(CELL_TYPES),
        "n_cells": n_cells,
        "sums": np.rint(sums).astype(np.int64),
        "min_counts": np.array([res["min_counts"] for res in scanpy["results"]]),
    }
    if not np.array_equal(arrays["sums"], sums):
        sys.exit("scanpy's sums are not whole numbers")
    for k, result in enumerate(scanpy["results"]):
        arrays[f"setting{k}__used"] = np.array(result["samples"]["used"])
        arrays[f"setting{k}__mouse"] = np.array(result["samples"]["mouse"])
        arrays[f"setting{k}__cell_type"] = np.array(result["samples"]["cell_type"])
        for column in (
            "n_samples_used",
            "n_samples_left_out",
            "n_cells_min",
            "total_counts_min",
        ):
            arrays[f"setting{k}__{column}"] = np.array(
                [np.nan if g[column] is None else g[column] for g in result["groups"]],
                dtype=float,
            )
        arrays[f"setting{k}__group_cell_type"] = np.array(
            [g["cell_type"] for g in result["groups"]]
        )
        arrays[f"setting{k}__group_dose"] = np.array(
            [g["dose"] for g in result["groups"]]
        )
    np.savez_compressed(FIXTURE_DIRECTORY / "scanpy_counts.npz", **arrays)
    aliasing = [
        {**case, "r": found}
        for case, found in zip(cases, r["aliasing"], strict=True)
        if not case["name"].startswith("TCDD")
    ]
    tcdd = next(case for case in cases if case["name"].startswith("TCDD"))
    aliasing.insert(0, {**tcdd, "r": r["aliasing"][cases.index(tcdd)]})
    (FIXTURE_DIRECTORY / "r_aliasing.json").write_text(
        json.dumps({"versions": r["versions"], "cases": aliasing}, indent=1) + "\n"
    )
    minimums = [
        {**design, "r": found}
        for design, found in zip(designs, r["minimums"], strict=True)
    ]
    (FIXTURE_DIRECTORY / "etap.json").write_text(
        json.dumps({"versions": r["versions"], "designs": minimums}, indent=1) + "\n"
    )
    np.savez_compressed(FIXTURE_DIRECTORY / "gse208161.npz", **gse)
    print(
        f"wrote scanpy_counts.npz, r_aliasing.json, etap.json and gse208161.npz to {FIXTURE_DIRECTORY}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help="the directory of the GEO file")
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    arguments = parser.parse_args()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = Path(
        arguments.data
        or os.environ.get("GSE208161_DATA")
        or cache / "cytepool-validation" / "GSE208161"
    ).expanduser()
    if directory.resolve().is_relative_to(FIXTURE_DIRECTORY.parents[2]):
        sys.exit(f"{directory} is in the repository; keep the data outside it")
    series = tcdd_series()
    cells = simulate_cells(series)
    n_cells, totals = cells_and_totals(cells, series)
    settings = [0.0, stellate_threshold(n_cells, totals)]
    cube = counts_cube(cells, series)
    scanpy, scanpy_versions = run_scanpy(cells, series, settings)
    cases = aliasing_cases(series, n_cells, totals)
    designs = etap_designs(series, n_cells, totals)
    r, r_versions = run_r(cases, designs)
    gse_rows, gse = gse208161(read_metadata(fetch(directory)))
    print(f"Reference: {scanpy_versions}; {r_versions}")
    rows = [
        *compare_counts(cube, series, scanpy),
        *compare_aliasing(cases, r["aliasing"], cube),
        *compare_minimums(designs, r["minimums"]),
        *gse_rows,
    ]
    failures = sum(row[-1] == "FAIL" for row in rows)
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if arguments.write_fixtures:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(series, n_cells, scanpy, cases, r, designs, gse)
    print(f"{len(rows)} checks, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
