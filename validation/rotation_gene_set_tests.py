"""Check the rotation gene-set tests roast, mroast and romer against limma.

Compares ``cytepool.rotation_gene_set_tests`` with limma's ``roast``, ``mroast``
and ``romer`` on log-expression of 1,500 genes simulated from fixed seeds, with gene
sets that change up, down, in both directions or not at all, share random sample
effects, overlap, hold one to five genes or 40% of the genes, or carry contribution
weights (signs, sizes, zeros). The cases cover a contrast given as a design column or
as weights, observation, voom and sample weights, a trended prior estimated robustly,
an infinite prior df (also with very large effects) and weights for every gene.

- Exact: CytePool replays the standard normal draws R gives limma after ``set.seed``
  (roast's 1,000 rotations at a time, romer's in one block; found by probing), so
  p-values, active proportions, numbers of genes and directions must agree to 1e-12.
  limma tests each set with the same seed, as CytePool's mroast shares its rotations.
  Deliberate differences have rows of their own, and their explanations are checked
  where they can be: where the observed floormean or msq statistic is the smallest it
  can be, CytePool's p-value must be 1; floormean's mixed p-value with negative weights
  must be limma's with the absolute weights; and romer without shrinkage must agree
  exactly where limma's posterior probabilities are NaN (log-odds beyond 709.8).
- Calibrated: each program draws its own B rotations, for mroast and romer over every
  set: each difference must be within 4 sd(p_1 - p_2) + 2 / (B + 1), with sd from the
  binomial variance at the mean of the two. mroast's rule for adjusted p-values
  (Benjamini-Hochberg of the mid-p-values, raised to at least the p-values) is checked
  on limma's own output.
- fry: mroast with 20,000 rotations against fry, the limit of its mean statistic with
  equal variances, and fry's mixed p-values of sets of 2 to d genes against the msq
  rotation test of the same statistic, for CytePool's fry and limma's.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that ``cytepool``
    imports from the working tree.
    R 4.4 or newer with the Bioconductor package limma and jsonlite. Set ``RSCRIPT`` to
    the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/rotation_gene_set_tests.py
    PYTHONPATH=. python validation/rotation_gene_set_tests.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool import Design, fry, mroast, roast, romer
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.voom import voom

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "rotation_gene_set_tests"
)
EXACT = 1e-12
FRY_MIXED = 0.1  # the beta approximation's error, up to about 0.07 (gene_set_tests.py)
FLOOR = 0.6744897501960817  # floormean's mixed floor, the root of chi-squared's median
ROAST = ("mean", "floormean", "mean50", "msq")
R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(jsonlite)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(dir, name) {
  dims <- scan(file.path(dir, paste0(name, ".dim")), quiet = TRUE)
  v <- readBin(file.path(dir, paste0(name, ".bin")), "double", n = prod(dims), size = 8)
  if (length(dims) == 2) matrix(v, dims[1], dims[2]) else v
}
for (case in fromJSON(file.path(root, "cases.json"), simplifyVector = FALSE)) {
  dir <- file.path(root, case$name); y <- rd(dir, "y"); X <- rd(dir, "X")
  w <- if (file.exists(file.path(dir, "w.bin"))) rd(dir, "w") else NULL
  contrast <- unlist(case$contrast); k <- ncol(y) - ncol(X) + 1
  X2 <- if (length(contrast) == 1) X[, c(setdiff(seq_len(ncol(X)), contrast), contrast)] else
    contrastAsCoef(X, contrast, first = FALSE)$design
  rows <- lapply(case$sets, function(s) unlist(s$rows) + 1)
  gw <- lapply(case$sets, function(s) if (is.null(s$w)) NULL else unlist(s$w))
  index <- lapply(seq_along(rows), function(i)
    if (is.null(gw[[i]])) rows[[i]] else data.frame(rows[[i]], gw[[i]]))
  out <- list()
  for (run in case$runs) {
    B <- run$nrot; draws <- file.path(dir, paste0(run$name, ".draws"))
    args <- list(design = X, contrast = contrast, nrot = B, set.statistic = run$stat,
                 weights = w, trend = isTRUE(run$trend), robust = isTRUE(run$robust))
    one <- function(i, weights) {
      set.seed(run$seed)
      do.call(roast, c(list(y = y, index = rows[[i]], gene.weights = weights), args))
    }
    set.seed(run$seed)
    if (run$test == "roast") {  # the rotations of each set, drawn 1,000 at a time
      parts <- list(); left <- B
      while (left > 0) {
        b <- min(1000, left); left <- left - b
        parts[[length(parts) + 1]] <- t(matrix(rnorm(b * k), b, k))
      }
      writeBin(as.vector(do.call(cbind, parts)), draws, size = 8)
      sets <- lapply(seq_along(rows), function(i) {
        r <- one(i, gw[[i]])
        set.seed(run$seed)
        m <- do.call(mroast, c(list(y = y, index = index[i]), args))
        res <- list(p = r$p.value$P.Value, active = r$p.value$Active.Prop,
                    n = m$NGenes, up = m$Direction == "Up", two_sided = m$PValue)
        if (any(gw[[i]] < 0)) res$abs_weights <- one(i, abs(gw[[i]]))$p.value$P.Value
        if (isTRUE(run$scaled) && !is.null(gw[[i]]))
          res$scaled <- one(i, 10 * gw[[i]])$p.value$P.Value
        res
      })
      # limma's z-scores of the moderated t-statistics, on the df roast uses
      fit <- eBayes(lmFit(y, X2, weights = w), trend = isTRUE(run$trend),
                    robust = isTRUE(run$robust))
      df <- pmin(fit$df.prior + fit$df.residual, 1e4)
      res <- list(sets = sets, z = zscoreT(fit$t[, ncol(X2)], df, approx = TRUE))
    } else if (run$test == "romer") {
      writeBin(rnorm(B * k), draws, size = 8)
      set.seed(run$seed)
      r <- romer(y, rows, X, contrast = contrast, nrot = B, set.statistic = run$stat,
                 shrink.resid = !isFALSE(run$shrink), array.weights = w)
      res <- list(up = r[, "Up"], down = r[, "Down"], mixed = r[, "Mixed"], n = r[, "NGenes"])
      fit <- eBayes(lmFit(y, X2, weights = w))
      fit <- eBayes(lmFit(y, X2, weights = w), proportion = 1 - propTrueNull(fit$p.value[, ncol(X2)]))
      res$overflow <- sum(fit$lods[, ncol(X2)] > log(.Machine$double.xmax))
    } else if (run$test == "mroast") {
      res <- as.list(do.call(mroast, c(list(y = y, index = index, sort = "none"), args)))
    } else {
      f <- fry(y, rows, X, contrast = contrast, standardize = "none", sort = "none")
      res <- list(mixed = f$PValue.Mixed)
    }
    out[[run$name]] <- res
  }
  write_json(out, file.path(dir, "limma.json"), digits = NA)
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma"))),
           file.path(root, "versions.txt"))
"""


class Replay(np.random.Generator):
    """A generator whose standard normal draws are given: R's, for limma's rotations."""

    def __init__(self, draws):
        super().__init__(np.random.PCG64())
        self.draws, self.used = np.asarray(draws, dtype=float).ravel(), 0

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        n = int(np.prod(size))
        self.used += n
        return self.draws[self.used - n : self.used].reshape(size)


def simulate(rng, x, n_genes, column, *, equal=False, strong=False):
    """Genes x samples log-expression and gene sets (positions, contribution weights)."""
    n, p = x.shape
    mean = rng.uniform(3.0, 11.0, n_genes)
    sd = np.sqrt(0.08 * 4.0 / rng.chisquare(4.0, n_genes)) * np.exp(-0.12 * (mean - 7))
    noise = rng.normal(size=(n_genes, n))
    if equal:  # the same residual sum of squares in every gene
        q = np.linalg.qr(x, mode="complete")[0]
        residual = noise @ q[:, p:]
        residual *= np.sqrt((n - p) / np.sum(residual**2, axis=1))[:, None]
        noise = noise @ q[:, :p] @ q[:, :p].T + residual @ q[:, p:].T
        sd = np.full(n_genes, 0.3)
    y = mean[:, None] + noise * sd[:, None]
    shift = x[:, column] - x[:, column].mean()
    for start, stop, size in (
        (0, 30, 1.2),
        (30, 55, -1.0),
        (85, 125, 1.2),
        (125, 135, -1.2),
    ):
        y[start:stop] += size * sd[start:stop, None] * shift
    if strong:  # |t| from 10 to 50, with log-odds beyond 709.8 for the largest
        y[:60] += rng.uniform(8.0, 40.0, 60)[:, None] * sd[:60, None] * shift
    if not equal:
        y[55:115] += 0.8 * sd[55:115, None] * rng.normal(size=n)
    sets = {
        "up": (range(0, 30), None), "down": (range(30, 55), None),
        "correlated": (range(55, 85), None), "correlated_up": (range(85, 115), None),
        "mixed": (range(115, 135), None), "overlap": ([*range(20, 40), *range(150, 170)], None),
        "single": ([140], None), "pair": ([141, 142], None), "five": (range(143, 148), None),
        "null": (sorted(rng.choice(np.arange(170, n_genes), 30, replace=False)), None),
        "large": (range(int(0.6 * n_genes), n_genes), None),
        "signature": ([*range(0, 8), *range(30, 38)], [1.0] * 8 + [-1.0] * 8),
        "sizes": (range(115, 125), [0.5, 2, 1, 1, 3, 0.2, 1, 1, 1, 1]),
        "partly_zero": ([*range(0, 8), *range(200, 208)], [1.0] * 8 + [0.0] * 8),
    }  # fmt: skip
    return y, {k: (np.asarray(list(r)), w) for k, (r, w) in sets.items()}


def runs(test, statistics, seed, nrot=999, suffix="", **options):
    return [dict(name=f"{test}_{s}{suffix}", test=test, stat=s, nrot=nrot, seed=seed + i,
                 **options) for i, s in enumerate(statistics)]  # fmt: skip


def calibrated(test, stat, seed, nrot):
    return [{"name": f"{test} own rotations {stat}", "test": test, "stat": stat,
             "nrot": nrot, "seed": seed}]  # fmt: skip


def unweighted(sets):
    return {k: v for k, v in sets.items() if v[1] is None}


def make_cases(rng, n_genes):
    """Each case: data, design, contrast, sets and the runs of both programs."""
    cases = []

    def add(name, design, y, sets, contrast, case_runs, weights=None):
        cases.append({"name": name, "design": design, "y": y, "sets": sets,
                      "contrast": contrast, "runs": case_runs, "weights": weights})  # fmt: skip

    groups = Design({"group": np.repeat(["A", "B", "C"], 3)}, "~ group")
    y, sets = simulate(rng, groups.matrix, n_genes, 1)
    add("groups", groups, y, sets, "groupB",
        runs("roast", ROAST, 10) + calibrated("mroast", "mean", 30, 1999)
        + calibrated("mroast", "msq", 31, 1999))  # fmt: skip
    add("groups_romer", groups, y, unweighted(sets), "groupB",
        runs("romer", ROAST[:3], 20) + runs("romer", ["mean"], 24, suffix="_unshrunk", shrink=False)
        + calibrated("romer", "mean", 32, 999) + calibrated("romer", "mean50", 33, 999))  # fmt: skip
    interleaved = Design({"group": np.tile(["A", "B", "C"], 3)}, "~ group",
                         contrasts={"c_vs_b": "groupC - groupB"})  # fmt: skip
    y, sets = simulate(rng, interleaved.matrix, n_genes, 2)
    add("groups_weights", interleaved, y, sets, "c_vs_b", runs("roast", ROAST[::2], 40),
        weights=np.exp(rng.normal(0.0, 0.5, (9, n_genes))))  # fmt: skip
    add("groups_trend_robust", interleaved, y, sets, "c_vs_b",
        runs("roast", ROAST[:2], 45, trend=True, robust=True))  # fmt: skip
    design = Design({"condition": np.tile(["ctrl", "stim"], 6),
                     "age": np.round(rng.uniform(20.0, 70.0, 12))}, "~ condition + age")  # fmt: skip
    expression, sets = simulate(rng, design.matrix, n_genes, 1)
    counts = rng.poisson(np.exp2(expression.T) * rng.uniform(0.5, 2.0, 12)[:, None])
    voomed = voom(counts, design)
    add("voom", design, voomed.log2_cpm.T, sets, "conditionstim",
        runs("roast", ["mean", "msq"], 50) + calibrated("mroast", "mean50", 52, 1999),
        weights=voomed.weights)  # fmt: skip
    paired = Design({"donor": np.repeat([f"D{i}" for i in range(6)], 2),
                     "treated": np.tile(["no", "yes"], 6)}, "~ donor + treated")  # fmt: skip
    y, sets = simulate(rng, paired.matrix, n_genes, 6)
    add("paired_sample_weights", paired, y, unweighted(sets), "treatedyes",
        runs("roast", ["mean"], 60) + runs("romer", ["mean"], 61),
        weights=rng.uniform(0.3, 3.0, 12))  # fmt: skip
    for name, strong, seed in (
        ("equal_variances", False, 70),
        ("strong_effects", True, 75),
    ):
        y, sets = simulate(rng, groups.matrix, n_genes, 1, equal=True, strong=strong)
        extra = [{"name": "fry", "test": "fry", "nrot": 0, "seed": 0}]
        if strong:  # without shrinkage, which the overflow does not reach
            extra = runs("romer", ["mean"], seed + 3, suffix="_unshrunk", shrink=False)
        add(name, groups, y, unweighted(sets), "groupB",
            runs("roast", ROAST[::2], seed) + runs("romer", ["mean"], seed + 2) + extra)  # fmt: skip
    y, sets = simulate(rng, groups.matrix, n_genes, 1)
    gene_weights = rng.choice([-2.0, -1.0, 0.5, 1.0, 3.0], n_genes)
    sets = {k: (rows, gene_weights[rows].tolist()) for k, (rows, w) in sets.items()
            if w is None and k != "large"}  # fmt: skip
    add("gene_weights", groups, y, sets, "groupC",
        runs("roast", ROAST, 80) + runs("roast", ["msq"], 85, 1999, "_scaled", scaled=True))  # fmt: skip
    return cases


def run_limma(cases):
    with tempfile.TemporaryDirectory() as temporary:
        root, specs = Path(temporary), []
        for case in cases:
            directory = root / case["name"]
            directory.mkdir()
            design, contrast = case["design"], case["contrast"]
            arrays = {"y": case["y"], "X": np.asarray(design.matrix)}
            if case["weights"] is not None:
                weights = np.asarray(case["weights"])
                arrays["w"] = weights if weights.ndim == 1 else weights.T
            for key, value in arrays.items():
                (directory / f"{key}.bin").write_bytes(value.tobytes(order="F"))
                (directory / f"{key}.dim").write_text(" ".join(map(str, value.shape)))
            if contrast in design.columns:
                contrast = design.columns.index(contrast) + 1
            else:
                contrast = design.contrasts[contrast].tolist()
            sets = {name: {"rows": rows.tolist(), "w": w}
                    for name, (rows, w) in case["sets"].items()}  # fmt: skip
            specs.append({"name": case["name"], "contrast": contrast, "sets": sets,
                          "runs": case["runs"]})  # fmt: skip
        (root / "cases.json").write_text(json.dumps(specs))
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        for case in cases:
            directory = root / case["name"]
            case["limma"] = json.loads((directory / "limma.json").read_text())
            case["draws"] = {path.stem: np.fromfile(path, dtype="<f8")
                             for path in directory.glob("*.draws")}  # fmt: skip
        return (root / "versions.txt").read_text().split("\n")[:2]


def gene_set(rows, w):
    return rows.tolist() if w is None else dict(zip(rows.tolist(), w, strict=True))


def common(case):
    return {"data": case["y"].T, "design": case["design"], "contrast": case["contrast"],
            "weights": case["weights"]}  # fmt: skip


def limma_rows(ours):
    """roast's row as limma's four rows (down, up, up or down, mixed): the p-values
    and the active proportions, that of up or down the larger of down and up."""
    columns = ("down_p_value", "up_p_value", "p_value", "mixed_p_value")
    down, up, mixed = (ours[f"proportion_{c}"][0] for c in ("down", "up", "mixed"))
    p = np.array([ours[column][0] for column in columns], dtype=float)
    return p, np.array([down, up, max(down, up), mixed])


def expected_roast(stat, z, rows, w, reference):
    """limma's p-values, with CytePool's value where a deliberate difference is
    explained, and which entries are not compared at all."""
    a = np.ones(rows.size) if w is None else np.asarray(w, float)
    zs = z[rows[a != 0]]
    x, expected = a[a != 0] * zs, np.asarray(reference["p"], float).copy()
    compared, minimum = np.ones(4, dtype=bool), np.zeros(4, dtype=bool)
    if stat in ("floormean", "msq"):  # the observed statistic at its smallest value
        minimum[:2] = np.all(x >= 0.0), np.all(x <= 0.0)
        minimum[3] = stat == "floormean" and bool(np.all(np.abs(zs) <= FLOOR))
    if stat == "floormean" and np.any(a < 0):  # mixed: limma's, with |a|
        expected[3] = reference["abs_weights"][3]
    if stat == "msq" and np.any(a < 0) and np.any(a > 0):
        compared[:3] = False
    if stat == "mean50" and np.any(a == 0):
        compared[:] = False
    minimum &= expected < 1.0
    expected[minimum] = 1.0
    return expected, compared, minimum


def check_roast(case, run, table):
    """CytePool given limma's rotations: roast for each set and mroast for all of them."""
    limma, draws = case["limma"][run["name"]], case["draws"][run["name"]]
    name, stat, z = f"{case['name']} {run['name']}", run["stat"], np.asarray(limma["z"])
    sets = {k: gene_set(rows, w) for k, (rows, w) in case["sets"].items()}
    options = {"set_statistic": stat, "n_rotations": run["nrot"],
               "trend": run.get("trend", False), "robust": run.get("robust", False)}  # fmt: skip
    if run.get("scaled"):  # every weight multiplied by 10, with 1,999 rotations
        change = max(
            np.max(np.abs(np.subtract(r["scaled"], r["p"]))) for r in limma["sets"]
        )
        table.append((f"{name}: limma's p-values with the weights multiplied by 10",
                      "roast", "max absolute change", float(change), None))  # fmt: skip
        scaled = {k: {g: 10.0 * a for g, a in v.items()} for k, v in sets.items()}
        ours = [mroast(**common(case), gene_sets=s, rng=1, **options)
                for s in (sets, scaled)]  # fmt: skip
        change = max(
            np.max(np.abs(ours[0][c] - ours[1][c]))
            for c in ("p_value", "mixed_p_value")
        )
        table.append((f"{name}: CytePool's, the same", "roast", "max absolute change",
                      float(change), EXACT))  # fmt: skip
        return
    worst, deliberate = 0.0, {"minimum": [], "signs": [], "negative": [], "zero": []}
    for genes, (rows, w), reference in zip(
        sets.values(), case["sets"].values(), limma["sets"], strict=True
    ):
        ours = roast(**common(case), gene_set=genes, rng=Replay(draws), **options)
        (p, active), ref = limma_rows(ours), np.asarray(reference["p"], float)
        expected, compared, minimum = expected_roast(stat, z, rows, w, reference)
        worst = max(worst, float(np.max(np.abs(p - expected)[compared], initial=0.0)),
                    float(np.max(np.abs(active - reference["active"]))))  # fmt: skip
        deliberate["minimum"] += np.abs(p - ref)[minimum & compared].tolist()
        if stat == "msq" and not compared[0]:
            deliberate["signs"].append(float(np.max(np.abs(p - ref)[:3])))
        if stat == "floormean" and w is not None and min(w) < 0:
            deliberate["negative"].append(float(abs(p[3] - ref[3])))
        if w is not None and 0 in w:  # limma counts the genes of weight zero
            if not case.setdefault("zero reported", False):
                case["zero reported"] = True
                table.append((f"{case['name']}: genes of the set with weights of zero",
                              "roast", "CytePool, limma", f"{ours['n_genes'][0]}, "
                              f"{reference['n'][0]}", None))  # fmt: skip
            deliberate["zero"] += np.abs(p - ref)[~compared].tolist()
        else:
            worst = max(worst, float(abs(ours["n_genes"][0] - reference["n"][0])))
    table.append((f"{name}: p-values, active proportions, genes", "roast",
                  "max absolute difference", worst, EXACT))  # fmt: skip
    for key, what in (("minimum", "p-values at the smallest statistic"),
                      ("signs", "directional p-values of sets with weights of both signs"),
                      ("negative", "mixed p-values of sets with negative weights"),
                      ("zero", "mean50 p-values of the set with weights of zero")):  # fmt: skip
        if deliberate[key]:
            table.append((f"{name}: {what}", "roast", "max absolute difference",
                          float(max(deliberate[key])), None))  # fmt: skip
    ours = mroast(**common(case), gene_sets=sets, rng=Replay(draws), **options)
    # Not compared: msq with weights of both signs, mean50 with weights of zero.
    free = [w is None or not ((stat == "msq" and min(w) < 0 < max(w))
                              or (stat == "mean50" and 0 in w))
            for _, w in case["sets"].values()]  # fmt: skip
    directions = np.array([d == "up" for d in ours["direction"]])
    mismatched = int(np.sum((directions != [r["up"][0] for r in limma["sets"]])[free]))
    two_sided = np.abs(ours["p_value"] - [r["two_sided"][0] for r in limma["sets"]])[
        free
    ]
    table.append((f"{name}: mroast's directions, all sets sharing the rotations",
                  "mroast", "mismatched sets", mismatched, 0))  # fmt: skip
    table.append((f"{name}: mroast's two-sided p-values", "mroast",
                  "max absolute difference", float(np.max(two_sided, initial=0.0)),
                  EXACT))  # fmt: skip


def check_romer(case, run, table):
    limma, draws = case["limma"][run["name"]], case["draws"][run["name"]]
    sets = {k: rows for k, (rows, _) in case["sets"].items()}
    shrink = run.get("shrink", True)
    ours = romer(**common(case), gene_sets=sets, set_statistic=run["stat"],
                 n_rotations=run["nrot"], rng=Replay(draws), shrink_contrast_effects=shrink)  # fmt: skip
    difference = max(float(np.max(np.abs(ours[f"{c}_p_value"] - np.asarray(limma[c]))))
                     for c in ("up", "down", "mixed"))  # fmt: skip
    difference = max(difference, float(np.max(np.abs(ours["n_genes"] - limma["n"]))))
    label = f"{case['name']} {run['name']}"
    if shrink and limma["overflow"][0]:  # limma's posterior probabilities are NaN
        table.append((f"{label}: genes whose limma log-odds exceed 709.8", "romer",
                      "genes", int(limma["overflow"][0]), "info"))  # fmt: skip
        table.append((f"{label}: p-values", "romer", "max absolute difference",
                      difference, None))  # fmt: skip
    else:
        table.append((f"{label}: p-values and genes", "romer", "max absolute difference",
                      difference, EXACT))  # fmt: skip


def bound(ours, theirs, n):
    """4 sd(p_1 - p_2) + 2 / (n + 1) for two Monte Carlo p-values of n draws each."""
    p = np.clip((ours + theirs) / 2.0, 1.0 / (n + 1), 1.0)
    return 4.0 * np.sqrt(2.0 * p * (1.0 - p) / n) + 2.0 / (n + 1)


def check_calibrated(case, run, table):
    """CytePool's own rotations against limma's, as the ratio |p1 - p2| / bound."""
    limma, B, label = case["limma"][run["name"]], run["nrot"], case["name"]
    if run["test"] == "romer":
        sets = {k: rows for k, (rows, _) in case["sets"].items()}
        ours = romer(**common(case), gene_sets=sets, set_statistic=run["stat"],
                     n_rotations=B, rng=run["seed"])  # fmt: skip
        pairs = [(ours[f"{c}_p_value"], np.asarray(limma[c], float))
                 for c in ("up", "down", "mixed")]  # fmt: skip
    else:
        sets = {k: gene_set(rows, w) for k, (rows, w) in case["sets"].items()}
        ours = mroast(**common(case), gene_sets=sets, set_statistic=run["stat"],
                      n_rotations=B, rng=run["seed"])  # fmt: skip
        # Deliberate differences: msq with contribution weights (with more than 1,000
        # rotations limma's depends on their scale), and the genes and mean50 of the
        # set with weights of zero.
        both = np.array([w is not None and run["stat"] == "msq"
                         for _, w in case["sets"].values()])  # fmt: skip
        zero = np.array([w is not None and 0 in w for _, w in case["sets"].values()])
        free = ~zero | (run["stat"] != "mean50")
        free &= ~both
        pairs = [(ours["p_value"][free], np.asarray(limma["PValue"], float)[free]),
                 (ours["mixed_p_value"][free], np.asarray(limma["PValue.Mixed"], float)[free])]  # fmt: skip
        for p_column, fdr in (("PValue", "FDR"), ("PValue.Mixed", "FDR.Mixed")):
            p = np.asarray(limma[p_column], float)
            rule = np.maximum(adjust_p_values(p - 0.5 / (B + 1), "bh"), p)
            difference = float(np.max(np.abs(rule - np.asarray(limma[fdr], float))))
            table.append((f"{label} {run['stat']}: limma's {fdr} from its {p_column} by "
                          "mroast's rule", "mroast", "max absolute difference", difference,
                          EXACT))  # fmt: skip
        exact = [np.abs(ours["n_genes"] - limma["NGenes"])[~zero],
                 np.abs(ours["proportion_up"] - np.asarray(limma["PropUp"], float)),
                 np.abs(ours["proportion_down"] - np.asarray(limma["PropDown"], float))]  # fmt: skip
        table.append((f"{label} {run['name']}: genes and active proportions", "mroast",
                      "max absolute difference", float(max(map(np.max, exact))), EXACT))  # fmt: skip
    ratio = max(float(np.max(np.abs(a - b) / bound(a, b, B))) for a, b in pairs)
    table.append((f"{label} {run['name']}, B = {B}", run["test"],
                  "max abs difference / bound", ratio, 1.0))  # fmt: skip


def check_fry(case, table):
    """fry against the limit it computes, and its mixed p-values of small sets against
    the msq rotation test of the same statistic."""
    sets, B = {k: rows for k, (rows, _) in case["sets"].items()}, 20_000
    limit = fry(**common(case), gene_sets=sets, standardize="none")
    mean = mroast(**common(case), gene_sets=sets, n_rotations=B, rng=1)
    ratio = float(np.max(np.abs(mean["p_value"] - limit["p_value"])
                         / bound(mean["p_value"], limit["p_value"], B)))  # fmt: skip
    table.append((f"{case['name']}: mroast's mean, B = {B}, against fry", "fry",
                  "max abs difference / bound", ratio, 1.0))  # fmt: skip
    d = case["y"].shape[1] - np.asarray(case["design"].matrix).shape[1]
    small = (limit["n_genes"] >= 2) & (limit["n_genes"] <= d)
    msq = mroast(**common(case), gene_sets=sets, set_statistic="msq", n_rotations=B,
                 rng=2)["mixed_p_value"][small]  # fmt: skip
    limma = np.asarray(case["limma"]["fry"]["mixed"], float)[small]
    for label, values, tolerance in (("CytePool's", limit["mixed_p_value"][small],
                                      FRY_MIXED), ("limma's", limma, "info")):  # fmt: skip
        table.append((f"{case['name']}: {label} fry, mixed p-values of the sets of 2 to "
                      f"{d} genes, against msq, B = {B}", "mroast",
                      "max absolute difference", float(np.max(np.abs(values - msq))),
                      tolerance))  # fmt: skip


def compare(cases, table):
    for case in cases:
        for run in case["runs"]:
            if run["test"] == "fry":
                check_fry(case, table)
            elif "own rotations" in run["name"]:
                check_calibrated(case, run, table)
            elif run["test"] == "romer":
                check_romer(case, run, table)
            else:
                check_roast(case, run, table)


def write_fixtures(cases):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for case in cases:
        design, contrast = case["design"], case["contrast"]
        arrays = {"y": case["y"].T, "design": np.asarray(design.matrix),
                  "columns": np.array(design.columns), "contrast": np.array(contrast),
                  "runs": np.array(json.dumps(case["runs"]))}  # fmt: skip
        if contrast not in design.columns:
            arrays["contrast weights"] = design.contrasts[contrast]
        if case["weights"] is not None:
            arrays["weights"] = np.asarray(case["weights"])
        for name, (rows, w) in case["sets"].items():
            arrays[f"set {name}"] = rows
            if w is not None:
                arrays[f"set weights {name}"] = np.asarray(w)
        for run in case["runs"]:
            arrays[f"draws {run['name']}"] = case["draws"][run["name"]]
            arrays[f"limma {run['name']}"] = np.array(
                json.dumps(case["limma"][run["name"]])
            )
        np.savez_compressed(FIXTURE_DIRECTORY / f"{case['name']}.npz", **arrays)
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(1), 1500)
    fixtures = []
    if arguments.write_fixtures:  # 240 genes and 99 rotations, replayed by the tests
        kept = ("groups", "groups_romer", "groups_weights", "groups_trend_robust",
                "paired_sample_weights", "equal_variances", "gene_weights")  # fmt: skip
        fixtures = [
            c for c in make_cases(np.random.default_rng(2), 240) if c["name"] in kept
        ]
        for case in fixtures:
            case["name"] = f"fixture_{case['name']}"
            case["runs"] = [dict(run, nrot=99) for run in case["runs"]
                            if run["test"] in ("roast", "romer") and not run.get("scaled")
                            and "own rotations" not in run["name"]]  # fmt: skip
    versions = run_limma(cases + fixtures)
    print("Reference: " + ", ".join(versions))
    table = []
    compare(cases + fixtures, table)
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
            shown = f"{tolerance:.0e}" if 0 < tolerance < 0.1 else f"{tolerance:g}"
        failures += result == "FAIL"
        value = value if isinstance(value, (int, str)) else f"{value:.1e}"
        print(f"| {check} | {source} | {metric} | {value} | {shown} | {result} |")
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            for case in fixtures:
                case["name"] = case["name"].removeprefix("fixture_")
            write_fixtures(fixtures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
