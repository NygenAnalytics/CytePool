"""Check the gene-set tests camera and fry against limma.

Compares ``cytepool.stats.gene_set_tests`` with limma's ``camera`` and ``fry`` on
log-expression simulated from fixed seeds, with gene variances from a scaled inverse
chi-squared prior around a mean-variance trend. The sets change up, down or in both
directions, or not at all, share random sample effects (inter-gene correlation) with
or without a change, overlap others, hold one, two or five genes, or 40% of the genes.
The cases: three groups with the contrast as a design column; the same with only sets
of one or more than d genes (so that fry's adjusted mixed p-values compare); three
groups in interleaved order with observation weights and the contrast as weights (a
named contrast); counts with voom weights from ``voom`` (limma gets the same weights),
with the contrast both ways; a paired design with sample weights; equal residual
variances (infinite prior df); 16 samples with sets of 2 to 30 unchanged genes and
three sets with changed genes; a gene without residual variation and ten identical
genes; and 8 genes, fewer than the residual df. Designs are ``Design`` objects and
genes are named by id here; R gets the design matrices and the genes' positions.

camera runs with the default inter-gene correlation of 0.01, with it estimated (also
with negative values allowed), with ranks, with a trend and robustly. limma's camera
has no robust option: its reference is ``cameraPR`` (with an estimated correlation,
``rankSumTestWithCorrelation``, or ``cameraPR`` converted to the df of the estimate)
on limma's z-scores, ``zscoreT(approx = TRUE, method = "hill")`` of the moderated t of
``eBayes(robust = TRUE)``, fitted with the design of ``contrastAsCoef``. fry runs with
each standardization, and with a trend and robustly.

A row covers one case and run: the largest difference over the numbers of genes and
the directions (a mismatch is an infinite difference), the correlations (differences)
and the p-values and adjusted p-values (relative to max(p, 1e-300) max(1, |log p|), as
in ``empirical_bayes.py``; limma's p-values above 1 count as 1). Tiers: exact (1e-10);
close (1e-6) for robust moderation, whose prior df limma solves only to a tolerance;
and 1e-3 for fry's ``"posterior_sd"``, where limma's null expectation of the trimmed
mean square is about 7.3e-5 (d + 1) / d above the exact one. Deliberate differences,
one row per case: fry's mixed p-values of sets of 2 to d genes, which limma computes
in too few dimensions; a simulation of the rotation test checks CytePool's and shows
how far limma's are. Information: three groups sorted by group with observation
weights and the contrast as weights, where a pivot of every gene's QR decomposition is
zero in exact arithmetic and R's ``qr()`` signs some of them by rounding noise.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor package limma. Set ``RSCRIPT`` to the Rscript
    executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/gene_set_tests.py
    PYTHONPATH=. python validation/gene_set_tests.py --write-fixtures
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool.design import Design
from cytepool.stats.gene_set_tests import camera, fry
from cytepool.stats.voom import voom

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "gene_set_tests"
)
EXACT = 1e-10
CLOSE = 1e-6
# fry's posterior_sd: limma's null expectation of the trimmed mean square is about
# 7.3e-5 (d + 1) / d above the exact value CytePool uses, which moves p-values by about
# 1e-4 of their value.
POSTERIOR = 1e-3
# Mixed p-values against the simulated rotation test: the error of the beta
# approximation itself at moderate p-values, up to about 0.07 in these simulations
# (limma's is the same for sets of more than d genes), and simulation error.
ROTATION = 0.1
SCALED = "max scaled difference"
ABSOLUTE = "max absolute difference"
# Run name: (test, inter-gene correlation, ranks, negative allowed, trend, standardize,
# robust).
RUNS = {
    "camera": ("camera", "0.01", 0, 0, 0, "", 0),
    "camera_estimated": ("camera", "NA", 0, 0, 0, "", 0),
    "camera_negative": ("camera", "NA", 0, 1, 0, "", 0),
    "camera_ranks": ("camera", "0.01", 1, 0, 0, "", 0),
    "camera_ranks_estimated": ("camera", "NA", 1, 0, 0, "", 0),
    "camera_trend": ("camera", "0.01", 0, 0, 1, "", 0),
    "camera_robust": ("camera", "0.01", 0, 0, 0, "", 1),
    "camera_robust_estimated": ("camera", "NA", 0, 0, 1, "", 1),
    "camera_robust_ranks": ("camera", "NA", 1, 0, 0, "", 1),
    "fry_none": ("fry", "", 0, 0, 0, "none", 0),
    "fry_residual": ("fry", "", 0, 0, 0, "residual.sd", 0),
    "fry_posterior": ("fry", "", 0, 0, 0, "posterior.sd", 0),
    "fry_posterior_trend": ("fry", "", 0, 0, 1, "posterior.sd", 0),
    "fry_posterior_robust": ("fry", "", 0, 0, 0, "posterior.sd", 1),
}
COLUMNS = {
    "NGenes": "n_genes", "Correlation": "correlation", "Direction": "direction",
    "PValue": "p_value", "FDR": "adjusted_p_value", "PValue.Mixed": "mixed_p_value",
    "FDR.Mixed": "adjusted_mixed_p_value",
}  # fmt: skip

R_SCRIPT = r"""
suppressPackageStartupMessages(library(limma))
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(dir, name) {
  dims <- scan(file.path(dir, paste0(name, ".dim")), quiet = TRUE)
  v <- readBin(file.path(dir, paste0(name, ".bin")), "double", n = prod(dims), size = 8)
  if (length(dims) == 2) matrix(v, dims[1], dims[2]) else v
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  y <- rd(dir, "y"); X <- rd(dir, "X")
  w <- if (file.exists(file.path(dir, "w.bin"))) rd(dir, "w") else NULL
  lines <- strsplit(readLines(file.path(dir, "sets.txt")), "\t")
  index <- lapply(lines, function(l) as.integer(strsplit(l[2], ",")[[1]]))
  names(index) <- vapply(lines, function(l) l[1], "")
  spec <- readLines(file.path(dir, "contrast.txt"))
  column <- spec[1] == "column"
  contrast <- if (column) as.integer(spec[2]) else as.numeric(strsplit(spec[2], ",")[[1]])
  runs <- read.table(file.path(dir, "runs.txt"), header = TRUE, colClasses = "character",
                     na.strings = character(0))
  if (file.exists(file.path(dir, "pivots"))) {  # R's qr() signs of the last pivots
    X2 <- contrastAsCoef(X, contrast, first = FALSE)$design
    pivots <- sapply(seq_len(nrow(y)), function(g) qr(X2 * sqrt(w[g, ]))$qr[ncol(X), ncol(X)])
    writeBin(as.double(pivots), file.path(dir, "out_pivots R.bin"), size = 8)
  }
  G <- nrow(y); p <- ncol(X); df <- min(ncol(y) - p, G - 2)  # of an estimated correlation
  for (i in seq_len(nrow(runs))) {
    r <- runs[i, ]; cor <- if (r$cor == "NA") NA else as.numeric(r$cor)
    if (r$test == "fry") {
      out <- fry(y, index, X, contrast = contrast, weights = w, standardize = r$standardize,
                 trend = r$trend == "1", robust = r$robust == "1", sort = "none")
    } else if (r$robust == "0") {
      out <- camera(y, index, X, contrast = contrast, weights = w, inter.gene.cor = cor,
                    allow.neg.cor = r$neg == "1", use.ranks = r$ranks == "1",
                    trend.var = r$trend == "1", sort = FALSE)
    } else {
      X2 <- if (column) X[, c(setdiff(seq_len(p), contrast), contrast)] else
        contrastAsCoef(X, contrast, first = FALSE)$design
      eb <- eBayes(lmFit(y, X2, weights = w), trend = r$trend == "1", robust = TRUE)
      z <- zscoreT(eb$t[, p], df = eb$df.total, approx = TRUE, method = "hill")
      if (!is.na(cor)) {
        out <- cameraPR(z, index, use.ranks = r$ranks == "1", inter.gene.cor = cor, sort = FALSE)
      } else {
        rho <- camera(y, index, X, contrast = contrast, weights = w, inter.gene.cor = NA,
                      sort = FALSE)$Correlation
        used <- pmax(ifelse(is.na(rho), 0, rho), 0)
        pv <- numeric(length(index)); up <- logical(length(index))
        for (k in seq_along(index)) {
          if (r$ranks == "1") {
            tails <- rankSumTestWithCorrelation(index[[k]], z, correlation = used[k], df = df)
            pv[k] <- 2 * min(tails); up[k] <- tails[2] <= tails[1]
          } else {
            one <- cameraPR(z, index[k], inter.gene.cor = used[k], sort = FALSE)
            up[k] <- one$Direction == "Up"
            stat <- qt(one$PValue / 2, G - 2, lower.tail = FALSE)
            pv[k] <- 2 * pt(stat, df, lower.tail = FALSE)
          }
        }
        out <- data.frame(NGenes = lengths(index), Correlation = rho,
                          Direction = ifelse(up, "Up", "Down"), PValue = pv,
                          FDR = p.adjust(pv, "BH"))
      }
    }
    for (k in intersect(names(out), c("NGenes", "Correlation", "PValue", "FDR",
                                      "PValue.Mixed", "FDR.Mixed")))
      writeBin(as.double(out[[k]]), file.path(dir, paste0("out_", r$name, " ", k, ".bin")), size = 8)
    writeBin(ifelse(out$Direction == "Up", 1, -1),
             file.path(dir, paste0("out_", r$name, " Direction.bin")), size = 8)
  }
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma"))),
           file.path(root, "versions.txt"))
"""


def simulate(rng, x, n_genes, column, *, equal=False):
    """Genes x samples log-expression, and gene sets as positions.

    Variances follow a scaled inverse chi-squared prior (4 df) around a trend in the
    mean; ``equal`` gives every gene the same residual variance instead.
    """
    n, p = x.shape
    mean = rng.uniform(3.0, 11.0, n_genes)
    sd = np.sqrt(0.08 * 4.0 / rng.chisquare(4.0, n_genes)) * np.exp(-0.12 * (mean - 7))
    noise = rng.normal(size=(n_genes, n))
    if equal:
        # Noise whose part in the residual space has the same norm in every gene.
        q = np.linalg.qr(x, mode="complete")[0]
        residual = noise @ q[:, p:]
        residual *= np.sqrt((n - p) / np.sum(residual**2, axis=1))[:, None]
        noise = noise @ q[:, :p] @ q[:, :p].T + residual @ q[:, p:].T
        sd = np.full(n_genes, 0.3)
    y = mean[:, None] + noise * sd[:, None]
    sets = {
        "up": range(0, 30),
        "down": range(30, 55),
        "correlated": range(55, 85),
        "correlated_up": range(85, 115),
        "mixed": range(115, 135),
        "overlap": [*range(20, 40), *range(150, 170)],
        "single": [140],
        "pair": [141, 142],
        "five": range(143, 148),
        "null": sorted(rng.choice(np.arange(n_genes // 3, n_genes), 30, replace=False)),
        "large": range(int(0.6 * n_genes), n_genes),
    }
    sets = {name: np.asarray(list(rows), dtype=int) for name, rows in sets.items()}
    shift = x[:, column] - x[:, column].mean()
    for name, size in (("up", 1.2), ("down", -1.0), ("correlated_up", 0.8)):
        y[sets[name]] += size * sd[sets[name], None] * shift
    y[115:125] += 1.2 * sd[115:125, None] * shift
    y[125:135] -= 1.2 * sd[125:135, None] * shift
    for name in ("correlated", "correlated_up"):
        shared = rng.normal(size=n)
        if not equal:  # shared effects would make the residual variances unequal
            y[sets[name]] += 0.8 * sd[sets[name], None] * shared
    return y, sets


def make_cases(rng, n_genes):
    """The cases: inputs for both programs, the contrast and the runs."""
    cases = {}
    camera_runs = [name for name in RUNS if name.startswith("camera")]
    fry_runs = [name for name in RUNS if name.startswith("fry")]

    def add(name, design, y, sets, contrast, runs, *, weights=None, voom_result=None):
        cases[name] = {
            "name": name, "design": design, "y": y, "sets": sets,
            "contrast": contrast, "runs": runs, "weights": weights,
            "voom": voom_result,
        }  # fmt: skip

    groups = Design(
        {"group": np.repeat(["A", "B", "C"], 3)},
        "~ group",
        contrasts={"c_vs_b": "groupC - groupB"},
    )
    y, sets = simulate(rng, groups.matrix, n_genes, 1)
    add("groups", groups, y, sets, "groupB", camera_runs + fry_runs)
    # Sets of one gene or of more than d genes: every column of fry compares.
    large = {name: rows for name, rows in sets.items() if rows.size not in range(2, 7)}
    add("groups_large_sets", groups, y, large, "groupB", fry_runs)
    # With the samples sorted by group and C - B as weights, a pivot of every gene's
    # QR decomposition is zero in exact arithmetic, and R's qr() signs a few of them
    # by rounding noise: information only.
    weights = np.exp(rng.normal(0.0, 0.5, (9, n_genes)))
    add("sorted_groups_weights", groups, y, sets, "c_vs_b",
        ["camera_estimated", "fry_residual"], weights=weights)  # fmt: skip
    interleaved = Design(
        {"group": np.tile(["A", "B", "C"], 3)},
        "~ group",
        contrasts={"c_vs_b": "groupC - groupB"},
    )
    y, sets = simulate(rng, interleaved.matrix, n_genes, 1)
    add("groups_weights", interleaved, y, sets, "c_vs_b", camera_runs + fry_runs,
        weights=weights)  # fmt: skip

    samples = {
        "condition": np.tile(["ctrl", "stim"], 6),
        "age": np.round(rng.uniform(20.0, 70.0, 12)),
    }
    design = Design(samples, "~ condition + age", contrasts={"stim": "conditionstim"})
    expression, sets = simulate(rng, design.matrix, n_genes, 1)
    counts = rng.poisson(np.exp2(expression.T) * rng.uniform(0.5, 2.0, 12)[:, None])
    result = voom(counts, design)
    for name, contrast in (("voom", "conditionstim"), ("voom_weights", "stim")):
        add(name, design, result.log_cpm.T, sets, contrast, camera_runs + fry_runs,
            voom_result=result)  # fmt: skip

    paired = Design(
        {"donor": np.repeat([f"D{i}" for i in range(6)], 2),
         "treated": np.tile(["no", "yes"], 6)},
        "~ donor + treated",
    )  # fmt: skip
    y, sets = simulate(rng, paired.matrix, n_genes, 6)
    add("sample_weights", paired, y, sets, "treatedyes",
        ["camera", "camera_estimated", "camera_ranks", "fry_none", "fry_residual"],
        weights=rng.uniform(0.3, 3.0, 12))  # fmt: skip

    y, sets = simulate(rng, groups.matrix, n_genes, 1, equal=True)
    add("equal_variances", groups, y, sets, "groupB",
        ["camera", "camera_estimated", "fry_none", "fry_posterior"])  # fmt: skip

    # Sets of 2 to 30 genes without a change, from gene 150 on.
    many = Design({"group": np.repeat(["A", "B"], 8)}, "~ group")
    y, _ = simulate(rng, many.matrix, max(n_genes, 400), 1)
    sizes = range(2, 31, 2)
    starts = 150 + np.cumsum([0, *sizes])
    small = {
        f"genes_{m}": np.arange(s, s + m)
        for m, s in zip(sizes, starts[:-1], strict=True)
    }
    # Sets with changed genes, whose mixed p-values lie far in the tail.
    small |= {f"tail_{m}": np.arange(100, 100 + m) for m in (16, 20, 30)}
    add("small_sets", many, y, small, "groupB", ["fry_none", "fry_residual"])

    y, sets = simulate(rng, groups.matrix, n_genes, 1)
    fitted = groups.matrix @ np.linalg.lstsq(groups.matrix, y[140], rcond=None)[0]
    y[140] = fitted  # no residual variation
    y[160:170] = y[160]  # identical genes: an estimated correlation of 1
    sets |= {"with_constant": np.array([140, *range(55, 65)]),
             "identical": np.arange(160, 170)}  # fmt: skip
    add("zero_variance", groups, y, sets, "groupB",
        ["camera_estimated", "camera", "camera_ranks_estimated"])  # fmt: skip

    # Fewer genes than residual df: an estimated correlation has G - 2 df.
    two = Design({"group": np.repeat(["A", "B"], 7)}, "~ group")
    y = rng.normal(size=(8, 14)) * rng.uniform(0.5, 1.5, (8, 1))
    y[:3, 7:] += 1.0
    few = {"a": np.array([0, 1, 2]), "b": np.array([3, 4]), "c": np.array([1, 5, 6, 7])}
    add("few_genes", two, y, few, "groupB",
        ["camera", "camera_estimated", "camera_ranks_estimated"])  # fmt: skip
    return cases


def write_inputs(cases, root):
    for name, case in cases.items():
        directory = root / name
        directory.mkdir()
        arrays = {"y": case["y"], "X": np.asarray(case["design"].matrix)}
        if case["voom"] is not None:
            arrays["w"] = case["voom"].weights.T
        elif case["weights"] is not None:
            arrays["w"] = case["weights"].T  # genes x samples, or sample weights
        for key, value in arrays.items():
            (directory / f"{key}.bin").write_bytes(value.tobytes(order="F"))
            (directory / f"{key}.dim").write_text(" ".join(map(str, value.shape)))
        (directory / "sets.txt").write_text(
            "".join(
                f"{set_name}\t{','.join(str(i + 1) for i in rows)}\n"
                for set_name, rows in case["sets"].items()
            )
        )
        design, contrast = case["design"], case["contrast"]
        if contrast in design.columns:
            spec = f"column\n{design.columns.index(contrast) + 1}\n"
        else:
            weights = design.contrasts[contrast].tolist()
            spec = "weights\n" + ",".join(map(repr, weights)) + "\n"
        (directory / "contrast.txt").write_text(spec)
        if case["name"] == "sorted_groups_weights":
            (directory / "pivots").touch()
        lines = ["name test cor ranks neg trend standardize robust"]
        for run in case["runs"]:
            test, cor, ranks, neg, trend, standardize, robust = RUNS[run]
            lines.append(
                f"{run} {test} {cor or 'NA'} {ranks} {neg} {trend} "
                f"{standardize or 'none'} {robust}"
            )
        (directory / "runs.txt").write_text("\n".join(lines) + "\n")


def run_limma(cases):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_inputs(cases, root)
        (root / "cases.txt").write_text("\n".join(cases) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        for name, case in cases.items():
            case["limma"] = {
                path.stem[4:]: np.fromfile(path, dtype="<f8")
                for path in (root / name).glob("out_*.bin")
            }
        return (root / "versions.txt").read_text().split("\n")[:2]


def cytepool_results(case, run):
    test, cor, ranks, neg, trend, standardize, robust = RUNS[run]
    design = case["design"]
    genes = [f"g{i:04d}" for i in range(case["y"].shape[0])]
    sets = {name: [genes[i] for i in rows] for name, rows in case["sets"].items()}
    data = case["voom"] if case["voom"] is not None else case["y"].T
    common = {"gene_ids": genes, "weights": case["weights"], "trend": bool(trend)}
    if test == "fry":
        style = {"none": "none", "residual.sd": "residual_sd"}
        return fry(data, design, case["contrast"], sets, robust=bool(robust),
                   standardize=style.get(standardize, "posterior_sd"), **common)  # fmt: skip
    return camera(
        data, design, case["contrast"], sets, robust=bool(robust),
        inter_gene_correlation=None if cor == "NA" else float(cor),
        allow_negative_correlation=bool(neg), use_ranks=bool(ranks), **common,
    )  # fmt: skip


def difference(ours, reference, kind):
    """Largest scaled difference; inf where NaN or the direction differ."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    keep = ~np.isnan(reference)
    if not np.array_equal(np.isnan(ours), ~keep):
        return np.inf
    if not keep.any():
        return 0.0
    ours, reference = ours[keep], reference[keep]
    if kind in ("NGenes", "Direction"):
        return 0.0 if np.array_equal(ours, reference) else np.inf
    if kind == "Correlation":
        return float(np.max(np.abs(ours - reference)))
    scale = np.maximum(np.abs(reference), 1e-300)
    scale *= np.maximum(1.0, np.abs(np.log(scale)))
    return float(np.max(np.abs(ours - reference) / scale))


def compared_columns(case, run, table):
    """CytePool's columns keyed by limma's names, with limma's values."""
    pairs = {}
    for key, values in case["limma"].items():
        name, column = key.split(" ")
        if name != run:
            continue
        ours = table[COLUMNS[column]]
        if column == "Direction":
            ours = np.array(
                [np.nan if v is None else 1 - 2 * (v == "down") for v in ours]
            )
            values = np.where(np.isnan(table["p_value"]), np.nan, values)
        elif column == "PValue":
            values = np.minimum(values, 1.0)  # limma's rank test can exceed 1
        pairs[column] = (ours, values)
    return pairs


def compare(cases, table):
    """One row per case and run, and one per case for each deliberate difference."""
    for name, case in cases.items():
        d = case["y"].shape[1] - case["design"].matrix.shape[1]
        sizes = np.array([rows.size for rows in case["sets"].values()])
        small = (sizes >= 2) & (sizes <= d)
        mixed_gap = []
        for run in case["runs"]:
            ours = cytepool_results(case, run)
            pairs = compared_columns(case, run, ours)
            if run.startswith("fry") and small.any():
                # The adjusted mixed p-values depend on those of every set.
                del pairs["FDR.Mixed"]
                ours_values, reference = pairs.pop("PValue.Mixed")
                pairs["PValue.Mixed"] = (ours_values[~small], reference[~small])
                mixed_gap.append(np.max(np.abs(ours_values - reference)[small]))
            differences = {k: difference(*pair, k) for k, pair in pairs.items()}
            worst = max(differences, key=differences.__getitem__)
            check = f"{name} {run}: {len(differences)} columns, largest {worst}"
            if case["name"] == "sorted_groups_weights":
                table.append(
                    (check, run.split("_")[0], SCALED, differences[worst], "info")
                )
            # With an infinite prior df both posterior standardizations are "none".
            elif run.startswith("fry_posterior") and case["name"] != "equal_variances":
                table.append((check, "fry", SCALED, differences[worst], POSTERIOR))
            elif "robust" in run:
                source = "cameraPR, rankSumTestWithCorrelation"
                table.append((check, source, SCALED, differences[worst], CLOSE))
            else:
                table.append(
                    (check, run.split("_")[0], SCALED, differences[worst], EXACT)
                )
        if "pivots R" in case["limma"]:
            # Every gene's last pivot is zero in exact arithmetic, which R's qr() signs
            # so that R's diagonal element is negative; a positive one is noise.
            noise = int(np.count_nonzero(case["limma"]["pivots R"] > 0))
            check = f"{name}: genes whose last pivot R's qr() signs by rounding noise"
            table.append((check, "qr", "genes", noise, "info"))
        if mixed_gap:
            check = f"{name}: fry's mixed p-values of the sets of 2 to {d} genes"
            table.append((check, "fry", ABSOLUTE, float(max(mixed_gap)), None))


def rotation_check(cases, table):
    """Mixed p-values against a simulation of the rotation test they approximate."""
    case = cases["small_sets"]
    x = np.asarray(case["design"].matrix)
    n, p = x.shape
    q = np.linalg.qr(x, mode="complete")[0]
    effects = (case["y"] @ q)[:, p - 1 :]  # contrast and residual effects, any signs
    rng = np.random.default_rng(20261025)
    rotations = rng.normal(size=(200_000, n - p + 1))
    rotations /= np.linalg.norm(rotations, axis=1)[:, None]
    ours = cytepool_results(case, "fry_none")["mixed_p_value"]
    limma = np.minimum(case["limma"]["fry_none PValue.Mixed"], 1.0)
    rows_of = {
        f"CytePool's, sets of 2 to {n - p} genes": [],
        f"limma's, sets of 2 to {n - p} genes": [],
        f"both programs', sets of {n - p + 1} to 30 genes": [],
        "both programs', sets with changed genes (relative)": [],
    }
    for i, (name, rows) in enumerate(case["sets"].items()):
        block = effects[rows]
        observed = np.sum(block[:, 0] ** 2)
        simulated = np.mean(np.sum((rotations @ block.T) ** 2, axis=1) >= observed)
        groups = list(rows_of.values())
        if name.startswith("tail"):
            groups[3].append(abs(ours[i] / simulated - 1.0))
        elif rows.size <= n - p:
            groups[0].append(abs(ours[i] - simulated))
            groups[1].append(abs(limma[i] - simulated))
        else:
            groups[2].append(abs(ours[i] - simulated))
    for label, values in rows_of.items():
        check = f"small_sets: {label} mixed p-values against 200,000 random rotations"
        tolerance = ROTATION if label.startswith("CytePool") else "info"
        metric = "max relative difference" if "relative" in label else ABSOLUTE
        table.append((check, "simulation", metric, float(np.max(values)), tolerance))


def write_fixtures(cases):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, case in cases.items():
        design = case["design"]
        # Samples x genes, as CytePool takes them, with the design and its columns.
        arrays = {
            "y": case["y"].T,
            "design": np.asarray(design.matrix),
            "columns": np.array(design.columns),
            "contrast": np.array(case["contrast"]),
            "runs": np.array(case["runs"]),
        }
        if case["contrast"] not in design.columns:
            arrays["contrast weights"] = design.contrasts[case["contrast"]]
        arrays |= {f"set {k}": v for k, v in case["sets"].items()}
        weights = case["weights"] if case["voom"] is None else case["voom"].weights
        if weights is not None:
            arrays["weights"] = weights
        arrays |= {f"limma {k}": v for k, v in case["limma"].items()}
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(1), 1500)
    small = make_cases(np.random.default_rng(2), 200)
    small = {f"fixture_{name}": case for name, case in small.items()}
    everything = cases | small if arguments.write_fixtures else cases
    versions = run_limma(everything)
    print("Reference: " + ", ".join(versions))
    table = []
    compare(everything, table)
    rotation_check(cases, table)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    failures = 0
    for check, source, metric, value, tolerance in table:
        if tolerance is None:
            result, shown = "deliberate difference", "n/a"
        elif tolerance == "info":
            result, shown = "information", "n/a"
        else:
            result = "pass" if value <= tolerance else "FAIL"
            shown = f"{tolerance:.0e}"
        failures += result == "FAIL"
        shown_value = value if isinstance(value, int) else f"{value:.1e}"
        print(f"| {check} | {source} | {metric} | {shown_value} | {shown} | {result} |")
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
