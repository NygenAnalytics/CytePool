"""Compare cytepool.normalization with edgeR and DESeq2.

Needs Python 3.13+ with numpy and numba, and R with the Bioconductor packages
edgeR and DESeq2. R runs as a separate program (``RSCRIPT``, default
``Rscript``); inputs and outputs travel as float64 binary files through a
temporary directory, so values are compared bit for bit.

Inputs come from fixed seeds: dense, sparse, very sparse (most upper quartiles
zero), unequal-depth, tie-heavy, large-group and non-integer count matrices;
300 small tie-heavy matrices for the default TMM reference and the trimming
boundaries; and hand-built edge cases. Every check pairs an R call with a
CytePool call. ``--write-fixtures`` saves the inputs, calls and R outputs of
the small cases to ``tests/data/normalization``.

Run from the repository root:

    RSCRIPT=/path/to/Rscript PYTHONPATH=. python validation/normalization.py
    RSCRIPT=/path/to/Rscript PYTHONPATH=. python validation/normalization.py --write-fixtures
"""

import argparse
import ast
import itertools
import json
import os
import subprocess
import sys
import tempfile
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

import cytepool.normalization as nz

TOLERANCE = 1e-10  # exact tier: relative difference, or absolute on the log2 scale
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "data" / "normalization"
FIXTURE_CASES = {"ties", "non-integer, small"}
VARIABLES = ("Y", "L", "g", "X", "X0", "nf", "ac", "mp")
TMM_KWARGS = "reference log_ratio_trim sum_trim weighted a_cutoff library_sizes".split()
METRICS = {"logcpm": "max abs diff of log2", "filter": "features that differ"}
METRICS["proportional"] = "max rel spread of the ratio"
# calls.json, which the tests read, names the function log2_cpm and the argument
# sample_groups of filter_by_expression as they were named when it was written.
FIXTURE_NAMES = {"log2_cpm": "log_cpm", "sample_groups": "group"}

# kind | check | R call | CytePool function and keyword arguments ($v: case variable v)
PER_DATASET = """
factor | upper quartile factors, p = 0.75 | calcNormFactors(Y, method='upperquartile') | upper_quartile_factors
factor | upper quartile factors, p = 0.9 | calcNormFactors(Y, method='upperquartile', p=0.9) | upper_quartile_factors p=0.9
factor | RLE factors | calcNormFactors(Y, method='RLE') | rle_factors
factor | RLE factors, library sizes given | calcNormFactors(Y, method='RLE', lib.size=L) | rle_factors library_sizes=$L
factor | DESeq2 ratio size factors | dds(Y, 'ratio') | median_ratio_size_factors
factor | DESeq2 poscounts size factors | dds(Y, 'poscounts') | median_ratio_size_factors method=poscounts
proportional | DESeq2 estimateSizeFactorsForMatrix poscounts, up to one factor | estimateSizeFactorsForMatrix(Y, type='poscounts') | median_ratio_size_factors method=poscounts
cpm | CPM | cpm(Y) | cpm
cpm | CPM, normalization factors | cpm(DGEList(Y, norm.factors=nf)) | cpm norm_factors=$nf
cpm | CPM, library sizes given | cpm(Y, lib.size=L) | cpm library_sizes=$L
logcpm | logCPM, prior count 2 | cpm(Y, log=TRUE, prior.count=2) | log2_cpm prior_count=2
logcpm | logCPM, prior count 0.5 | cpm(Y, log=TRUE, prior.count=0.5) | log2_cpm prior_count=0.5
logcpm | logCPM, prior count 5 | cpm(Y, log=TRUE, prior.count=5) | log2_cpm prior_count=5
logcpm | logCPM, prior count 2, normalization factors | cpm(DGEList(Y, norm.factors=nf), log=TRUE) | log2_cpm norm_factors=$nf
logcpm | logCPM, prior count 2, library sizes given | cpm(Y, lib.size=L, log=TRUE) | log2_cpm library_sizes=$L
filter | filterByExpr, group | filterByExpr(Y, group=factor(g)) | filter_by_expression sample_groups=$g
filter | filterByExpr, group, min.count 5, min.total.count 10 | filterByExpr(Y, group=factor(g), min.count=5, min.total.count=10) | filter_by_expression sample_groups=$g min_count=5 min_total_count=10
filter | filterByExpr, group, large.n 2, min.prop 0.5 | filterByExpr(Y, group=factor(g), large.n=2, min.prop=0.5) | filter_by_expression sample_groups=$g large_n=2 min_prop=0.5
filter | filterByExpr, design with a covariate | filterByExpr(Y, design=X) | filter_by_expression design=$X
filter | filterByExpr, design without an intercept | filterByExpr(Y, design=X0) | filter_by_expression design=$X0
filter | filterByExpr, no group or design (one group) | filterByExpr(Y) | filter_by_expression
filter | filterByExpr, normalization factors | filterByExpr(DGEList(Y, group=factor(g), norm.factors=nf)) | filter_by_expression sample_groups=$g norm_factors=$nf
"""
PER_SMALL_MATRIX = """
factor | TMM factors, default reference, 300 small tie-heavy matrices | calcNormFactors(Y, lib.size=L) | tmm_factors library_sizes=$L
factor | TMM factors, unweighted, 300 small tie-heavy matrices | calcNormFactors(Y, lib.size=L, doWeighting=FALSE) | tmm_factors library_sizes=$L weighted=False
factor | upper quartile factors, 300 small tie-heavy matrices | calcNormFactors(Y, lib.size=L, method='upperquartile') | upper_quartile_factors library_sizes=$L
factor | DESeq2 poscounts, 300 small tie-heavy matrices | dds(Y, 'poscounts') | median_ratio_size_factors method=poscounts
"""
EDGE_CASES = """
factor | empty trimmed set | calcNormFactors(Y) | tmm_factors
factor | no shared features | calcNormFactors(Y) | tmm_factors
factor | single sample | calcNormFactors(Y) | tmm_factors
factor | identical profiles, M below 1e-6 | calcNormFactors(Y, lib.size=L) | tmm_factors library_sizes=$L
factor | identical profiles, M above 1e-6 | calcNormFactors(Y, lib.size=L) | tmm_factors library_sizes=$L
factor | A cutoff equal to an A value | calcNormFactors(Y, Acutoff=ac) | tmm_factors a_cutoff=$ac
factor | A ties at the trimming boundary | calcNormFactors(Y, refColumn=1) | tmm_factors reference=0
factor | sparse counts: reference by square-root counts | calcNormFactors(Y) | tmm_factors
factor | median upper quartile 1e-20: documented reference | calcNormFactors(Y, lib.size=L) | tmm_factors library_sizes=$L
factor | median upper quartile below 1e-20: square-root reference | calcNormFactors(Y, lib.size=L) | tmm_factors library_sizes=$L
factor | identical above the A cutoff only | calcNormFactors(Y, lib.size=L, refColumn=2, Acutoff=ac) | tmm_factors library_sizes=$L reference=1 a_cutoff=$ac
filter | count at the CPM cutoff, L = 100036 | filterByExpr(Y, group=factor(g), lib.size=L) | filter_by_expression sample_groups=$g library_sizes=$L
filter | count at the CPM cutoff, L = 100015 | filterByExpr(Y, group=factor(g), lib.size=L) | filter_by_expression sample_groups=$g library_sizes=$L
filter | all-zero design, taken as an intercept | filterByExpr(Y, design=X) | filter_by_expression design=$X
filter | near-collinear design, eps = 1.5e-07 | filterByExpr(Y, design=X) | filter_by_expression design=$X
filter | near-collinear design, eps = 2.5e-07 | filterByExpr(Y, design=X) | filter_by_expression design=$X
filter | sample threshold 2 + 8.9e-15 | filterByExpr(Y, group=factor(g), large.n=0, min.prop=mp) | filter_by_expression sample_groups=$g large_n=0 min_prop=$mp
filter | sample threshold 2 + 1.1e-14 | filterByExpr(Y, group=factor(g), large.n=0, min.prop=mp) | filter_by_expression sample_groups=$g large_n=0 min_prop=$mp
"""
# Inputs that CytePool rejects, where R returns values (deviation) or fails too (error).
DEVIATIONS = """
deviation | all-zero sample: edgeR TMM with refColumn gives it factor 1 | calcNormFactors(Y, refColumn=1) | tmm_factors
deviation | upper quartile of zero: edgeR returns NaN factors | calcNormFactors(Y, method='upperquartile') | upper_quartile_factors
deviation | group and design both given: edgeR ignores the design | filterByExpr(Y, group=factor(g), design=X) | filter_by_expression sample_groups=$g design=$X
error | negative count: edgeR normLibSizes rejects it too | normLibSizes(Y) | tmm_factors
"""

R_DRIVER = r"""
suppressPackageStartupMessages({library(edgeR); library(DESeq2)})
dir <- commandArgs(trailingOnly = TRUE)[1]
calls <- read.delim(file.path(dir, "calls.tsv"), stringsAsFactors = FALSE, quote = "")
dds <- function(Y, type) {  # DESeq()'s route; non-integer counts only through the matrix route
  if (any(Y != round(Y))) return(estimateSizeFactorsForMatrix(Y, type = type))
  storage.mode(Y) <- "integer"
  d <- DESeqDataSetFromMatrix(Y, data.frame(s = seq_len(ncol(Y))), ~1)
  sizeFactors(estimateSizeFactors(d, type = type, quiet = TRUE))
}
load_case <- function(id) {
  d <- file.path(dir, id)
  shape <- scan(file.path(d, "shape.txt"), quiet = TRUE)
  e <- new.env()
  for (v in c("Y", "L", "g", "X", "X0", "nf", "ac", "mp")) {
    f <- file.path(d, paste0(v, ".bin"))
    if (file.exists(f)) e[[v]] <- readBin(f, "double", file.size(f) / 8)
  }
  e$Y <- matrix(e$Y, nrow = shape[2])  # samples x features in C order
  for (v in c("X", "X0")) if (!is.null(e[[v]])) e[[v]] <- matrix(e[[v]], nrow = shape[1])
  e
}
cases <- list()
for (i in seq_len(nrow(calls))) {
  id <- calls$case[i]
  if (is.null(cases[[id]])) cases[[id]] <- load_case(id)
  out <- tryCatch(as.double(suppressWarnings(suppressMessages(
    eval(parse(text = calls$expr[i]), envir = cases[[id]])))), error = function(e) NULL)
  if (!is.null(out)) writeBin(out, file.path(dir, sprintf("out_%d.bin", i)))
}
writeLines(c(R.version.string, paste("edgeR", packageVersion("edgeR")),
             paste("DESeq2", packageVersion("DESeq2"))), file.path(dir, "versions.txt"))
"""


def parse(table):
    """Rows of a check table as (kind, check, R call, CytePool call)."""
    rows = []
    for line in table.strip().splitlines():
        kind, check, r, call = (field.strip() for field in line.split("|"))
        function, *tokens = call.split()
        kwargs = {}
        for key, value in (token.split("=", 1) for token in tokens):
            try:
                kwargs[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                kwargs[key] = value  # a word, or a case variable
        rows.append((kind, check, r, dict(function=function, kwargs=kwargs)))
    return rows


def call(spec, case):
    kwargs = {}
    for key, value in spec["kwargs"].items():
        if isinstance(value, str) and value.startswith("$"):
            value = case[value[1:]]
            value = float(value[0]) if key in ("a_cutoff", "min_prop") else value
        kwargs[key] = value
    return getattr(nz, spec["function"])(case["Y"], **kwargs)


def matrix(text):
    """A float64 matrix from rows of numbers, one row per line."""
    return np.array([row.split() for row in text.strip().splitlines()], np.float64)


def simulate(seed, n_samples, n_features, depth, spread, integer=True):
    """Gamma-Poisson counts; 10% of features change fourfold in half the samples."""
    rng = np.random.default_rng(seed)
    share = rng.gamma(0.6, 1.0, n_features)
    share /= share.sum()
    log_spread = rng.uniform(np.log(spread[0]), np.log(spread[1]), n_samples)
    fold = np.ones((n_samples, n_features))
    fold[: n_samples // 2, rng.choice(n_features, n_features // 10, replace=False)] = 4
    expected = rng.gamma(5.0, depth * np.exp(log_spread)[:, None] * share * fold / 5)
    return (rng.poisson(expected) if integer else expected).astype(np.float64)


def datasets():
    rng = np.random.default_rng(2026)
    out = {}
    for name, seed, n, p, depth, spread, groups, integer in [
        ("dense", 1, 12, 2000, 2e6, (0.7, 1.4), [0] * 6 + [1] * 6, True),
        ("sparse", 2, 24, 2000, 2e4, (0.7, 1.4), [0] * 8 + [1] * 8 + [2] * 8, True),
        ("very sparse", 8, 20, 2000, 300, (0.5, 2.0), [0] * 10 + [1] * 10, True),
        ("unequal depth", 3, 8, 2000, 1e5, (0.14, 7.0), [0] * 4 + [1] * 4, True),
        ("ties", 4, 6, 400, 1500, (0.9, 1.1), [0, 0, 1, 1, 2, 2], True),
        ("large groups", 5, 40, 2000, 1e5, (0.7, 1.4), [0] * 20 + [1] * 20, True),
        ("non-integer", 6, 6, 2000, 1e5, (0.7, 1.4), [0] * 3 + [1] * 3, False),
        ("non-integer, small", 7, 6, 200, 1e4, (0.7, 1.4), [0] * 3 + [1] * 3, False),
    ]:
        y = simulate(seed, n, p, depth, spread, integer)
        y[:, :5] = 0.0  # all-zero features
        levels = sorted(set(groups))[1:]
        indicators = [[float(x == level) for x in groups] for level in levels]
        x = np.column_stack([np.ones(n), *indicators, rng.normal(size=n)])
        sizes = np.round(y.sum(axis=1) * rng.uniform(0.8, 1.25, n))
        if name == "ties":
            sizes = np.full(n, 1500.0)  # equal sizes tie the default reference
        case = dict(Y=y, g=np.array(groups, np.float64), X=x, X0=x[:, 1:], L=sizes)
        out[name] = case | dict(nf=nz.tmm_factors(y), integer=integer)
    return out


def alternative_references(y, sizes):
    """References chosen by plausible rules that differ from edgeR's in the last bits."""
    y = y[:, y.sum(axis=0) > 0]
    choices = set()  # numpy's quantile of y / N or of y, with numpy's or the exact mean
    by_ratio = np.quantile(y / sizes[:, None], 0.75, axis=1)
    for q in (by_ratio, np.quantile(y, 0.75, axis=1) / sizes):
        for mean in (q.mean(), float(sum(map(Fraction, q)) / q.size)):
            choices.add(int(np.argmin(np.abs(q - mean))))
    return choices


def small_matrices(n_random=200, n_sensitive=100):
    """Low integer counts: random matrices, a third with equal library sizes and
    a third sparse, then matrices whose TMM factors change under an alternative
    reference rule."""
    rng = np.random.default_rng(7)
    cases = {}
    while len(cases) < n_random + n_sensitive:
        sensitive = len(cases) >= n_random
        n = int(rng.integers(4, 9) if sensitive else rng.integers(2, 7))
        y = rng.integers(0, 15 if sensitive else 12, (n, int(rng.integers(5, 41))))
        # Every third random matrix is sparse, often with zero upper quartiles.
        if not sensitive and len(cases) % 3 == 1:
            y *= rng.uniform(size=y.shape) < rng.uniform(0.05, 0.5)
        if (y.sum(axis=1) == 0).any():
            continue
        equal = sensitive or len(cases) % 3 == 0
        sizes = np.full(n, float(rng.integers(50, 400))) if equal else y.sum(axis=1)
        y = y.astype(np.float64)
        if sensitive:
            default = nz.tmm_factors(y, library_sizes=sizes)
            other = alternative_references(y, sizes)
            factors = [
                nz.tmm_factors(y, library_sizes=sizes, reference=j) for j in other
            ]
            if max(np.max(np.abs(f / default - 1)) for f in factors) <= TOLERANCE:
                continue
        cases[f"small {len(cases)}"] = dict(Y=y, L=sizes, integer=True)
    return cases


def edge_cases():
    """Hand-built inputs for the documented and probed rules (see EDGE_CASES)."""
    same = np.array([[10, 20, 30, 40, 55, 70, 5, 8]] * 2, np.float64) * 100
    two = matrix("120 340 25 8 61 900 14 \n 150 300 31 3 70 800 9")
    a = (np.log2(two[0] / two[0].sum()) + np.log2(two[1] / two[1].sum())) / 2
    cases = {
        # M ties (a, a, b, b): no average rank lies in the kept window [2, 3].
        "empty trimmed set": dict(Y=matrix("1 2 3 6 \n 1 2 1 2")),
        "no shared features": dict(Y=matrix("5 0 3 0 \n 0 7 0 2")),
        "single sample": dict(Y=matrix("5 3 0 8")),
        "identical profiles, M below 1e-6": dict(Y=same, L=np.array([1000.00069, 1e3])),
        "identical profiles, M above 1e-6": dict(Y=same, L=np.array([1000.0007, 1e3])),
        "A cutoff equal to an A value": dict(Y=two, ac=np.array([np.sort(a)[2]])),
        # Ties in A that log2(p_k p_r) / 2 would break differently, moving the trim.
        "A ties at the trimming boundary": dict(
            Y=matrix("""
            25 5 2 14 15 23 26 18 2 28 1 29 4 27 28 18 28 24 9 17 13 22 20
            17 2 1 26 6 5 16 10 3 16 2 10 28 1 5 24 16 19 6 28 19 0 16
            3 24 13 22 29 1 23 24 17 3 7 4 29 13 13 19 15 21 28 16 29 15 25""")
        ),
        # Two upper quartiles of zero make the median zero.
        "sparse counts: reference by square-root counts": dict(
            Y=matrix("""
            0 0 0 16 0 0 0 26 0 \n 22 0 0 0 11 0 0 0 0 \n 0 12 8 21 0 26 11 0 2""")
        ),
    }
    # Upper quartiles 7.25, 1 and 31.25 over library sizes that put the median
    # ratio at 1e-20 exactly, or just below; the two rules pick samples 1 and 2.
    y = matrix("3 7 2 9 4 6 5 8 \n 1 1 1 1 1 1 0 2 \n 20 35 12 40 28 16 30 25")
    for name, size in (
        ("1e-20: documented", 1e20),
        ("below 1e-20: square-root", 1e20 * (1 + 1e-15)),
    ):
        sizes = np.array([7.25e25, size, 31.25e15])
        cases[f"median upper quartile {name} reference"] = dict(Y=y, L=sizes)
    # Log ratios of -1.4e-7 above the A cutoff and about -2.3 below it: the
    # samples count as identical only if the cutoff is applied first.
    y = np.column_stack([same, [[1, 1, 2], [5, 6, 9]]])
    sizes = np.array([y[1].sum() * (1 + 1e-7), y[1].sum()])
    a = (np.log2(y[0] / sizes[0]) + np.log2(y[1] / sizes[1])) / 2
    cutoff = np.array([(a[:8].min() + a[8:].max()) / 2])
    cases["identical above the A cutoff only"] = dict(Y=y, L=sizes, ac=cutoff)
    # One count at the CPM cutoff, where 10 / L * 1e6 rounds above (100036)
    # or below (100015) 1e6 * 10 / L.
    y = matrix("0 1000 1000 1000 \n 10 1000 1000 1000 \n 5 1000 1000 1000")
    for size in (100036, 100015):
        sizes = np.array([size - 5e4, size, size + 9e4])
        cases[f"count at the CPM cutoff, L = {size}"] = dict(
            Y=y, g=np.arange(3.0), L=sizes
        )
    # An all-zero design leaves the intercept that R's hat() adds: 6 samples.
    y = matrix(
        "20 20 20 20 20 0 \n 20 20 20 20 20 20 \n 1000 1000 1000 1000 1000 1000"
    ).T
    cases["all-zero design, taken as an intercept"] = dict(Y=y, X=np.zeros((6, 1)))
    # A third column g + eps w, which R's qr() keeps when eps / 2 > 1e-7.
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    w = np.array([1, -1, 0, 0, 1, -1, 0, 0.0]) / 2
    y = matrix("50 0 50 0 50 0 0 0 \n" + "1000 " * 8 + "\n" + "1000 " * 8).T
    for eps in (1.5e-7, 2.5e-7):
        x = np.column_stack([np.ones(8), g, g + eps * w])
        cases[f"near-collinear design, eps = {eps:g}"] = dict(Y=y, X=x)
    # min.prop sets the number of samples needed 8.9e-15 or 1.1e-14 above 2.
    y = matrix("20 20 0 0 0 0 \n 20 20 20 0 0 0 \n 1000 1000 1000 1000 1000 1000").T
    for delta in (8.9e-15, 1.1e-14):
        groups, prop = np.repeat([0.0, 1.0], 3), np.array([(2 + delta) / 3])
        cases[f"sample threshold 2 + {delta:g}"] = dict(Y=y, g=groups, mp=prop)
    return cases


def deviation_cases():
    """Inputs of DEVIATIONS, in its order."""
    y = matrix("5 3 8 2 6 \n 0 0 0 0 7 \n 3 4 1 9 2")
    inputs = [
        dict(Y=y * (np.arange(3) != 1)[:, None]),
        dict(Y=y),
        dict(Y=y + 1, g=np.arange(3.0), X=np.ones((3, 1))),
        dict(Y=y - 9 * (y == 8)),
    ]
    return {row[1]: case for row, case in zip(parse(DEVIATIONS), inputs)}


def checks(data, small):
    """(kind, check, case, R call, CytePool call) for every comparison."""
    rows = []
    for name, case in data.items():
        for ref, trims, weighted, cutoff, sizes in itertools.product(
            (None, 0), ((0.3, 0.05), (0.2, 0.1)), (True, False), (-1e10, -13.0), (0, 1)
        ):
            r = f"calcNormFactors(Y, logratioTrim={trims[0]}, sumTrim={trims[1]}, "
            r += f"doWeighting={str(weighted).upper()}, Acutoff={cutoff}"
            r += (", refColumn=1" if ref == 0 else "") + (
                ", lib.size=L" if sizes else ""
            )
            values = (ref, *trims, weighted, cutoff, "$L" if sizes else None)
            spec = dict(function="tmm_factors", kwargs=dict(zip(TMM_KWARGS, values)))
            rows.append(("factor", "TMM factors, 32 option sets", name, r + ")", spec))
        for kind, check, r, spec in parse(PER_DATASET):
            if "dds(Y, 'poscounts')" not in r or case["integer"]:
                rows.append((kind, check, name, r, spec))
    for name in small:
        rows.extend(
            (k, check, name, r, s) for k, check, r, s in parse(PER_SMALL_MATRIX)
        )
    for kind, name, r, spec in parse(EDGE_CASES):
        rows.append((kind, f"edge case: {name}", name, r, spec))
    return rows + [(k, name, name, r, s) for k, name, r, s in parse(DEVIATIONS)]


def run_reference(cases, rows):
    ids = {name: f"case{i}" for i, name in enumerate(cases)}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, case in cases.items():
            d = directory / ids[name]
            d.mkdir()
            (d / "shape.txt").write_text("%d %d\n" % case["Y"].shape)
            for v in (v for v in VARIABLES if v in case):
                # C order of samples x features is R's genes x samples.
                order = "F" if v in ("X", "X0") else "C"
                np.asarray(case[v], np.float64).ravel(order).tofile(d / f"{v}.bin")
        lines = ["case\texpr"] + [f"{ids[row[2]]}\t{row[3]}" for row in rows]
        (directory / "calls.tsv").write_text("\n".join(lines) + "\n")
        (directory / "driver.R").write_text(R_DRIVER)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(directory / "driver.R"), tmp], check=True
        )
        versions = (directory / "versions.txt").read_text().splitlines()
        outputs = [directory / f"out_{i}.bin" for i in range(1, len(rows) + 1)]
        results = [np.fromfile(f) if f.exists() else None for f in outputs]
    return versions, results


def compare(kind, ours, ref):
    """(difference, identical values, values), or None when both are undefined."""
    if ref is None or (kind == "factor" and np.isnan(ref).any()):
        return None if ours is None else (np.inf, 0, 1)
    if ours is None:
        return (np.inf, 0, ref.size)
    ours = np.ravel(ours).astype(np.float64)
    equal = ours == ref
    same = int(np.sum(equal))
    if kind == "filter":
        return (float(ref.size - same), same, ref.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "proportional":  # identical up to one factor common to all samples
            diff = np.abs(ours / ref / (ours[0] / ref[0]) - 1)
        else:
            scale = np.ones_like(ref) if kind == "logcpm" else np.abs(ref)
            diff = np.where(equal, 0.0, np.abs(ours - ref) / scale)
    return (float(np.max(np.nan_to_num(diff, nan=np.inf))), same, ref.size)


def md(*cells):
    print("| " + " | ".join(str(c) for c in cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore", UserWarning)  # no sample_groups or design given
    data, small, edges = datasets(), small_matrices(), edge_cases()
    cases = {**data, **small, **edges, **deviation_cases()}
    rows = checks(data, small)
    versions, results = run_reference(cases, rows)
    table, deviations = {}, []
    for (kind, check, name, r, spec), ref in zip(rows, results):
        try:
            ours = call(spec, cases[name])
        except ValueError:
            ours = None
        if kind in ("deviation", "error"):
            deviations.append((kind, check, ours is None, ref is not None))
            continue
        outcome = compare(kind, ours, ref)
        empty = dict(kind=kind, worst=0.0, same=0, total=0, undefined=0, cases=set())
        e = table.setdefault(check, empty)
        e["cases"].add(name)
        e["undefined"] += outcome is None
        if outcome is not None:
            e["worst"] = max(e["worst"], outcome[0])
            e["same"], e["total"] = e["same"] + outcome[1], e["total"] + outcome[2]
    print("Reference versions: " + "; ".join(versions) + f"; numpy {np.__version__}\n")
    md("check", "reference", "metric", "value", "tolerance", "result")
    md(*["---"] * 6)
    failed = False
    for check, e in table.items():
        tolerance = 0 if e["kind"] == "filter" else TOLERANCE
        ok = e["worst"] <= tolerance and e["total"] > 0  # some values compared
        failed |= not ok
        value = f"{e['worst']:.2g}" if tolerance else str(int(e["worst"]))
        value += f" ({e['same']}/{e['total']} values identical"
        value += f"; {e['undefined']} undefined in both)" if e["undefined"] else ")"
        n_data = len(e["cases"] & data.keys())
        label = f"{check} ({n_data} datasets)" if n_data else check
        source = "DESeq2" if "DESeq2" in check else "edgeR"
        metric = METRICS.get(e["kind"], "max rel diff")
        md(label, source, metric, value, f"{tolerance:g}", "pass" if ok else "FAIL")
    # Control: plausible rules for the default reference that differ from
    # edgeR's in floating-point details must change some small matrices.
    worst, same, total, changed = 0.0, 0, 0, set()
    for (kind, check, name, r, spec), ref in zip(rows, results):
        if check.startswith("TMM factors, default reference"):
            y, sizes = cases[name]["Y"], cases[name]["L"]
            for alternative in alternative_references(y, sizes):
                ours = nz.tmm_factors(y, library_sizes=sizes, reference=alternative)
                diff, n_same, n = compare(kind, ours, ref)
                worst, same, total = max(worst, diff), same + n_same, total + n
                changed |= {name} if diff > TOLERANCE else set()
    failed |= worst <= TOLERANCE
    label = "control: default reference by 4 rules that differ in the last bits, "
    label += "300 small matrices (100 chosen as sensitive)"
    value = (
        f"{worst:.2g} ({same}/{total} values identical; {len(changed)} matrices change)"
    )
    result = "control" if worst > TOLERANCE else "FAIL"
    md(label, "edgeR", "max rel diff", value, f"must exceed {TOLERANCE:g}", result)
    for kind, label, raised, r_returns in deviations:
        ok = raised and r_returns == (kind == "deviation")
        failed |= not ok
        result = ("deviation" if kind == "deviation" else "pass") if ok else "FAIL"
        r_text = "CytePool raises; R " + (
            "returns values" if kind == "deviation" else "fails"
        )
        md(
            f"{kind}: {label}",
            "edgeR",
            r_text,
            "raises" if raised else "no error",
            "n/a",
            result,
        )
    if args.write_fixtures:
        write_fixtures(cases, rows, results, small, edges)
    sys.exit(1 if failed else 0)


def write_fixtures(cases, rows, results, small, edges):
    """Inputs, calls and R outputs for the tests: the small datasets, the edge
    cases, 20 random small matrices, and TMM on 20 more and 30 sensitive ones.

    ``normalization.npz`` holds two flat float64 arrays; ``calls.json`` gives
    the offset and shape of every input and output in them.
    """
    stores, index, records = {"inputs": [], "outputs": []}, {}, []

    def place(name, values):
        values = np.asarray(values, np.float64)
        start = sum(v.size for v in stores[name])
        stores[name].append(values.ravel())
        return [start, list(values.shape)]

    for (kind, check, name, r, spec), ref in zip(rows, results):
        keep = name in FIXTURE_CASES or name in edges
        if name in small:
            i = int(name[6:])
            keep = i < 20 or ((i < 40 or 200 <= i < 230) and check.startswith("TMM"))
        skip = kind in ("proportional", "deviation", "error") or ref is None
        if not keep or skip or np.isnan(ref).any():  # R's NaN: CytePool raises
            continue
        case = cases[name]
        if name not in index:
            index[name] = {v: place("inputs", case[v]) for v in VARIABLES if v in case}
        if kind in ("cpm", "logcpm"):  # keep the first 40 features
            ref = ref.reshape(case["Y"].shape[0], -1)[:, :40]
        output = place("outputs", ref)
        recorded = dict(
            function=FIXTURE_NAMES.get(spec["function"], spec["function"]),
            kwargs={FIXTURE_NAMES.get(k, k): v for k, v in spec["kwargs"].items()},
        )
        records.append(
            dict(check=check, case=name, kind=kind, r_call=r, output=output, **recorded)
        )
    FIXTURES.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.concatenate(values) for name, values in stores.items()}
    np.savez_compressed(FIXTURES / "normalization.npz", **arrays)
    lines = ",\n".join(json.dumps(r) for r in records)
    text = '{"cases": ' + json.dumps(index) + ',\n"calls": [\n' + lines + "\n]}\n"
    (FIXTURES / "calls.json").write_text(text)


if __name__ == "__main__":
    main()
