"""Check the per-sample scores of gene sets (GSVA, ssGSEA, combined z-scores and PLAGE)
and their heatmap against GSVA.

Inputs, simulated from fixed seeds: log-expression values of 2,000 genes in 24
samples, 200 genes higher in the second 12, with two constant genes, a gene equal to
another, a gene of whole numbers (tied values) and a gene with an outlier sample;
negative binomial counts of the same genes in 16 samples, with means from e^2 to e^8;
Poisson counts with means from 0.3 to 6 in 12 samples (low counts); 400 genes in 210
samples; the log-expression values rounded to one decimal (ties); the log-expression
values with 2% of those of the first 12 samples missing; and a cube of one cell type
with the counts, whose samples have 3 to 60 cells. 66 gene sets of 1 to 300 genes, a
fifth of each outside the data: random sets, a set of one gene, one with the constant
genes, one that lists genes twice, two identical sets and a set of every gene (left
out for ssGSEA, whose normalization GSVA then cannot compute, but in one case). Each
case runs GSVA 2.0.0's ``gsva()`` with a ``gsvaParam``, ``ssgseaParam``,
``zscoreParam`` or ``plageParam`` and the CytePool function with the same options.

- Exact: the sets scored, NaN where GSVA gives NA, and every score within 1e-10 of
  the largest absolute score of the case: the GSVA method with each kernel, tau,
  maxDiff, absRanking and set sizes; ssGSEA with each alpha, with and without
  normalization, on tied values and with missing values (``use = "na.rm"``); combined
  z-scores, also with missing values; PLAGE up to the sign of each set; a cube's
  group, as log-CPM (edgeR's ``filterByExpr``, ``calcNormFactors`` and
  ``cpm(log = TRUE)``, which ``validation/normalization.py`` checks) and as counts;
  and the cells of ``plot_pathway_scores`` with ``standardize=True`` against GSVA's
  scores scaled as R's ``heatmap(scale = "row")`` scales them.
- Deliberate differences: the GSVA method in samples with missing values (GSVA's
  walk ends above zero there); PLAGE's sets of one gene (GSVA gives 1); and ssGSEA
  with a set of every gene (GSVA stops with an error).
- Information: the share of PLAGE's sets whose sign agrees with GSVA's, and the
  Poisson kernel on low counts, where genes whose kernel estimates are equal in exact
  arithmetic are ordered by the last bit of R's ``ppois``.

With ``--write-fixtures`` the cases on the log-expression values and the counts run on
300 genes and 21 sets, and the tests replay them. ``--benchmark`` times both programs
on 15,000 genes, 50 samples and 500 sets.

Requirements:
    Python 3.13 with numpy, numba and matplotlib, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with GSVA 2.0.0, BiocParallel and jsonlite (``RSCRIPT``, by default
    ``Rscript``).

Usage:
    PYTHONPATH=. python validation/pathway_scores.py
    PYTHONPATH=. python validation/pathway_scores.py --write-fixtures
    PYTHONPATH=. python validation/pathway_scores.py --benchmark
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.figure
import numpy as np

from cytepool.cube import Cube
from cytepool.normalization import filter_by_expression, log_cpm, tmm_factors
from cytepool.pathway_scores import combined_z_scores, gsva, plage, ssgsea
from cytepool.plotting import plot_pathway_scores

FIXTURES = Path(__file__).resolve().parents[1] / "tests/data/pathway_scores"
EXACT = 1e-10
FUNCTIONS = {
    "gsva": gsva,
    "ssgsea": ssgsea,
    "zscore": combined_z_scores,
    "plage": plage,
}
# GSVA's options, and CytePool's.
KERNELS = {"Gaussian": "gaussian", "Poisson": "poisson", "none": "none"}
NAMES = {"tau": "exponent", "alpha": "exponent", "normalize": "normalize",
         "minSize": "min_size", "maxSize": "max_size"}  # fmt: skip

R_SCRIPT = r"""
suppressPackageStartupMessages({library(GSVA); library(jsonlite); library(BiocParallel)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
input <- fromJSON(file.path(root, "input.json"), simplifyVector = FALSE)
matrices <- list()
for (name in names(input$data)) {
  d <- input$data[[name]]
  genes <- unlist(d$genes)
  x <- matrix(readBin(file.path(root, paste0(name, ".bin")), "double",
                      n = length(genes) * d$samples), length(genes), d$samples)
  x[is.na(x)] <- NA  # R's missing value, which GSVA recognises
  if (isTRUE(d$integer)) storage.mode(x) <- "integer"
  dimnames(x) <- list(genes, paste0("S", seq_len(d$samples)))
  matrices[[name]] <- x
}
param <- list(gsva = gsvaParam, ssgsea = ssgseaParam, zscore = zscoreParam,
              plage = plageParam)
score <- function(run) {
  p <- do.call(param[[run$method]], c(list(matrices[[run$data]],
                                           lapply(input$sets[[run$sets]], unlist)),
                                      run$options))
  suppressMessages(suppressWarnings(gsva(p, verbose = FALSE, BPPARAM = SerialParam())))
}
out <- list(cases = list())
for (case in input$cases) {
  out$cases[[case$name]] <- tryCatch({
    es <- score(case)
    # scale() of the samples x sets matrix is heatmap(scale = "row") of es.
    list(sets = rownames(es), scores = as.vector(t(es)), scaled = as.vector(scale(t(es))))
  }, error = function(e) list(error = conditionMessage(e)))
}
out$seconds <- sapply(input$benchmark, function(run) system.time(score(run))[["elapsed"]])
out$versions <- c(paste("R", getRversion()), paste("GSVA", packageVersion("GSVA")))
write_json(out, file.path(root, "output.json"), digits = I(17), na = "null",
           null = "null", auto_unbox = TRUE)
"""


# Inputs.


def log_expression(rng, n_genes, n_samples):
    """Gene means and noise, the first tenth of the genes higher in the second half of
    the samples."""
    y = rng.normal(0.0, 1.0, (n_samples, n_genes)) + rng.normal(5.0, 1.5, n_genes)
    y[n_samples // 2 :, : n_genes // 10] += rng.normal(1.0, 0.4, n_genes // 10)
    return y


def gene_sets(rng, genes, n_sets, largest):
    """Random sets of 1 to ``largest`` genes, a fifth outside the data, and the
    special sets."""
    sets = {}
    for s, size in enumerate(np.exp(rng.uniform(0.0, np.log(largest), n_sets))):
        members = list(rng.choice(genes, int(size), replace=False))
        members += [f"OUTSIDE{i}" for i in rng.choice(9999, int(size) // 5 + 1)]
        sets[f"set_{s:02d}"] = members
    sets["one_gene"] = [genes[25]]
    sets["constant_genes"] = [genes[3], genes[4], genes[5], genes[30], genes[31]]
    sets["listed_twice"] = genes[40:52] * 2
    sets["identical_a"] = sets["identical_b"] = genes[60:90:3]
    return sets


def cube_of(rng, genes, counts):
    """A cube of one cell type whose samples have 3 to 60 cells, one with no counts,
    and the log-CPM and counts GSVA's users pass for it: of its samples with at least
    10 cells and some counts, over the genes that filterByExpr keeps."""
    n_samples = counts.shape[0]
    pooled = counts.astype(np.int64)
    pooled[1] = 0
    cells = rng.integers(3, 60, n_samples)
    cells[0] = 4
    cube = Cube(
        {"counts": pooled[np.newaxis]},
        samples={"sample_id": [f"S{i + 1}" for i in range(n_samples)]},
        groups={"group_id": ["T cells"]},
        features={"feature_id": genes},
        n_cells=cells[:, np.newaxis],
    )
    kept = pooled[(pooled.sum(axis=1) > 0) & (cells >= 10)]
    keep = filter_by_expression(kept, group=np.zeros(kept.shape[0]))
    kept = kept[:, keep]
    values = log_cpm(kept, norm_factors=tmm_factors(kept), prior_count=2.0)
    return cube, values, kept, [g for g, k in zip(genes, keep, strict=True) if k]


def inputs(small):
    """The data (name to values, samples x genes, gene ids, stored as integers), the
    gene sets, the cases and the cube."""
    rng = np.random.default_rng(20261026)
    n_genes = 300 if small else 2000
    genes = [f"G{i:05d}" for i in range(n_genes)]
    y = log_expression(rng, n_genes, 10 if small else 24)
    y[:, 3] = 4.0  # constant genes, which every method but ssGSEA leaves out
    y[:, 4] = 7.5
    y[:, 6] = y[:, 7]  # two equal genes
    y[:, 8] = np.round(y[:, 8])  # tied values
    y[0, 9] = y[:, 9].max() + 6.0  # an outlier sample
    means = np.exp(rng.uniform(2.0, 8.0, n_genes))
    counts = rng.negative_binomial(8, 8 / (8 + means), (8 if small else 16, n_genes))
    missing = y.copy()  # 2% of the values of the first half of the samples missing
    missing[: y.shape[0] // 2][rng.random((y.shape[0] // 2, n_genes)) < 0.02] = np.nan
    data = {
        "log_expression": (y, genes, False),
        "counts": (counts, genes, True),
        "missing": (missing, genes, False),
    }
    sets = {"main": gene_sets(rng, genes, 15 if small else 60, 60 if small else 300)}
    sets["ranked"] = dict(sets["main"])
    sets["main"]["every_gene"] = list(genes)
    cases = [
        ("gsva, Gaussian kernel", "log_expression", "main", "gsva", {"kcdf": "Gaussian"}),
        ("gsva, tau 0.5, maxDiff FALSE", "log_expression", "main", "gsva",
         {"tau": 0.5, "maxDiff": False}),
        ("gsva, tau 2, absRanking", "log_expression", "main", "gsva",
         {"tau": 2.0, "absRanking": True}),
        ("gsva, no kernel", "log_expression", "main", "gsva", {"kcdf": "none"}),
        ("gsva, counts stored as integers (auto: Poisson)", "counts", "main", "gsva", {}),
        ("ssgsea", "log_expression", "ranked", "ssgsea", {}),
        ("ssgsea, alpha 1, not normalized", "log_expression", "ranked", "ssgsea",
         {"alpha": 1.0, "normalize": False}),
        ("zscore", "log_expression", "main", "zscore", {}),
        ("plage", "log_expression", "main", "plage", {}),
        ("gsva, missing values", "missing", "main", "gsva",
         {"kcdf": "Gaussian", "use": "na.rm"}),
        ("ssgsea, missing values", "missing", "ranked", "ssgsea", {"use": "na.rm"}),
        ("zscore, missing values", "missing", "main", "zscore", {}),
    ]  # fmt: skip
    cube = None
    if not small:
        low = rng.poisson(rng.uniform(0.3, 6.0, n_genes), (12, n_genes))
        many = log_expression(rng, 400, 210)
        sets["many"] = gene_sets(rng, genes[:400], 25, 120)
        cube, cube_values, cube_counts, kept = cube_of(rng, genes, counts)
        data |= {
            "decimals": (counts.astype(float), genes, False),
            "low": (low, genes, True),
            "ties": (np.round(y, 1), genes, False),
            "many": (many, genes[:400], False),
            "cube_log_cpm": (cube_values, kept, False),
            "cube_counts": (cube_counts, kept, True),
        }
        cases += [
            ("gsva, sizes 10 to 100", "log_expression", "main", "gsva",
             {"minSize": 10, "maxSize": 100}),
            ("gsva, counts stored as decimals (auto: Gaussian)", "decimals", "main",
             "gsva", {}),
            ("gsva, 210 samples (auto: no kernel)", "many", "many", "gsva", {}),
            ("gsva, low counts, Poisson kernel", "low", "main", "gsva",
             {"kcdf": "Poisson"}),
            ("ssgsea, alpha 0, tied values", "ties", "ranked", "ssgsea",
             {"alpha": 0.0}),
            ("ssgsea, a set of every gene", "log_expression", "main", "ssgsea", {}),
            ("gsva of a cube's group, log-CPM", "cube_log_cpm", "main", "gsva", {}),
            ("gsva of a cube's group, counts (auto: Poisson)", "cube_counts", "main",
             "gsva", {}),
        ]  # fmt: skip
    keys = ("name", "data", "sets", "method", "options")
    return data, sets, [dict(zip(keys, case, strict=True)) for case in cases], cube


def benchmark_inputs(rng):
    genes = [f"B{i:05d}" for i in range(15000)]
    means = np.exp(rng.uniform(2.0, 8.0, 15000))
    data = {
        "bench": (rng.normal(5.0, 1.0, (50, 15000)), genes, False),
        "bench_counts": (rng.negative_binomial(5, 5 / (5 + means), (50, 15000)),
                         genes, True),
    }  # fmt: skip
    sets = {f"set_{i}": list(rng.choice(genes, int(rng.integers(15, 500)), False))
            for i in range(500)}  # fmt: skip
    runs = [("gsva", "bench", {"kcdf": "Gaussian"}),
            ("gsva", "bench_counts", {"kcdf": "Poisson"}),
            ("ssgsea", "bench", {}), ("zscore", "bench", {}), ("plage", "bench", {})]  # fmt: skip
    keys = ("method", "data", "options")
    runs = [dict(zip(keys, run, strict=True), sets="bench") for run in runs]
    return data, {"bench": sets}, runs


def run_r(root, data, sets, cases, benchmark):
    description = {}
    for name, (values, genes, integer) in data.items():
        # A C-ordered samples x genes array is R's genes x samples matrix.
        (root / f"{name}.bin").write_bytes(np.asarray(values, float).tobytes())
        description[name] = {"genes": genes, "samples": values.shape[0],
                             "integer": integer}  # fmt: skip
    payload = {
        "data": description,
        "sets": sets,
        "cases": cases,
        "benchmark": benchmark,
    }
    (root / "input.json").write_text(json.dumps(payload))
    (root / "compare.R").write_text(R_SCRIPT)
    rscript = os.environ.get("RSCRIPT", "Rscript")
    command = f"ulimit -v 8000000; {rscript} --vanilla {root / 'compare.R'} {root}"
    subprocess.run(["bash", "-c", command], check=True)
    return json.loads((root / "output.json").read_text())


# CytePool's runs.


def options_of(case):
    """CytePool's options for GSVA's."""
    options = {}
    for key, value in case["options"].items():
        if key == "kcdf":
            options["kernel"] = KERNELS[value]
        elif key == "maxDiff":
            options["score_type"] = "max_deviation"
        elif key == "absRanking":
            options["score_type"] = "kuiper"
        elif key in NAMES:
            options[NAMES[key]] = value
    return options


def run_cytepool(case, data, sets, cube):
    function = FUNCTIONS[case["method"]]
    options = options_of(case)
    if case["data"].startswith("cube"):
        values = "counts" if case["data"] == "cube_counts" else "log_cpm"
        return function(cube, sets[case["sets"]], values=values, **options)
    values, genes, integer = data[case["data"]]
    values = values.astype(np.int64) if integer else values
    return function(values, sets[case["sets"]], gene_ids=genes, **options)


# Comparisons.


def matrix_of(scores, n_sets, n_samples):
    """GSVA's sets x samples scores, by row, as samples x sets."""
    values = np.array([np.nan if v is None else v for v in scores], dtype=float)
    return values.reshape(n_sets, n_samples).T


def difference(ours, theirs, nan_must_match=True):
    """The largest difference over the largest absolute value of GSVA's; inf when
    NaN stands in different places, unless ``nan_must_match`` is False, which
    compares where both are numbers."""
    if nan_must_match and not np.array_equal(np.isnan(ours), np.isnan(theirs)):
        return np.inf
    return float(np.nanmax(np.abs(ours - theirs)) / np.nanmax(np.abs(theirs)))


def check_case(case, reference, result, data, table):
    name = case["name"]
    if "error" in reference and name != "ssgsea, a set of every gene":
        table.append((f"{name}: GSVA stops ({reference['error'].strip()})", "GSVA",
                      "runs", np.inf, EXACT))  # fmt: skip
        return
    if "error" in reference:  # GSVA cannot normalize the scores
        column = list(result.gene_sets).index("every_gene")
        table.append((f"{name}: GSVA stops ({reference['error'].strip()}); CytePool "
                      "gives the set NaN and normalizes over the others", "GSVA",
                      "share of the set's scores that are NaN",
                      float(np.isnan(result.scores[:, column]).mean()), "deliberate"))  # fmt: skip
        return
    if list(result.gene_sets) != reference["sets"]:
        table.append((f"{name}: the sets scored", "GSVA", "the same", np.inf, EXACT))
        return
    ours = result.scores
    theirs = matrix_of(reference["scores"], len(reference["sets"]), ours.shape[0])
    label = f"{name}: {ours.shape[1]} sets x {ours.shape[0]} samples"
    metric = "max difference / max |score|"
    if case["method"] == "plage":
        several = result.n_genes > 1
        signs = np.sign(np.sum(ours * theirs, axis=0))[several]
        aligned = ours[:, several] * signs
        table.append((f"{label}: sets of 2 or more genes, up to sign", "GSVA", metric,
                      difference(aligned, theirs[:, several]), EXACT))  # fmt: skip
        table.append((f"{name}: sets whose sign is that of GSVA (R's svd)", "GSVA",
                      "share of the sets", float(np.mean(signs > 0)), "information"))  # fmt: skip
        table.append((f"{name}: the {int((~several).sum())} set(s) of one gene, "
                      "where GSVA gives 1 in every sample", "GSVA", metric,
                      difference(ours[:, ~several], theirs[:, ~several]),
                      "deliberate"))  # fmt: skip
        return
    if name == "gsva, missing values":
        complete = ~np.isnan(data[case["data"]][0]).any(axis=1)
        table.append((f"{label}: the {int(complete.sum())} samples without missing "
                      "values", "GSVA", metric,
                      difference(ours[complete], theirs[complete]), EXACT))  # fmt: skip
        table.append((f"{name}: the {int((~complete).sum())} samples with missing "
                      "values, the walk falling by 1/(p_j - k_j), where GSVA scores "
                      "the set of every gene", "GSVA", metric,
                      difference(ours[~complete], theirs[~complete], False),
                      "deliberate"))  # fmt: skip
        return
    tolerance = "information" if "low counts" in name else EXACT
    table.append((label, "GSVA", metric, difference(ours, theirs), tolerance))


def check_figure(reference, result, table):
    """The cells of the standardized heatmap, the samples grouped by condition."""
    groups = ["control", "treated"] * (result.scores.shape[0] // 2)
    ax = plot_pathway_scores(
        result, color_by=groups, standardize=True,
        ax=matplotlib.figure.Figure().subplots(),
    )  # fmt: skip
    cells = next(image for image in ax.get_images() if image.get_gid() == "scores")
    drawn = np.ma.filled(cells.get_array().astype(float), np.nan)
    order = np.argsort([group == "treated" for group in groups], kind="stable")
    scaled = matrix_of(reference["scaled"], *result.scores.shape[::-1])
    table.append((f"plot_pathway_scores: the {drawn.size:,} cells, standardized, "
                  "the samples grouped", 'GSVA, R\'s heatmap(scale = "row")',
                  "max difference / max |cell|", difference(drawn, scaled[order].T),
                  EXACT))  # fmt: skip


# Fixtures and benchmark.


def write_fixtures(data, sets, cases, output):
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    arrays = {name: np.asarray(values, float) for name, (values, _, _) in data.items()}
    scored = {}
    for case in cases:
        reference = output["cases"][case["name"]]
        n_samples = data[case["data"]][0].shape[0]
        n_sets = len(reference["sets"])
        arrays[case["name"]] = matrix_of(reference["scores"], n_sets, n_samples)
        scored[case["name"]] = reference["sets"]
    np.savez_compressed(FIXTURES / "gsva.npz", **arrays)
    fixture = {"genes": data["log_expression"][1], "sets": sets, "cases": cases,
               "scored": scored}  # fmt: skip
    (FIXTURES / "gsva.json").write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote the fixtures to {FIXTURES}")


def benchmark(data, sets, runs, seconds):
    print("| Method | GSVA, 1 thread (s) | CytePool (s) |")
    print("| --- | --- | --- |")
    for run, reference in zip(runs, seconds, strict=True):
        values, genes, integer = data[run["data"]]
        values = values.astype(np.int64) if integer else values
        function, options = FUNCTIONS[run["method"]], options_of(run)
        function(values[:4, :50], {"s": genes[:5]}, gene_ids=genes[:50], **options)
        best = np.inf
        for _ in range(3):
            start = time.perf_counter()
            function(values, sets["bench"], gene_ids=genes, **options)
            best = min(best, time.perf_counter() - start)
        label = f"{run['method']} {run['options'].get('kcdf', '')}".strip()
        print(f"| {label} | {reference:.2f} | {best:.2f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    arguments = parser.parse_args()
    if arguments.benchmark:
        data, sets, runs = benchmark_inputs(np.random.default_rng(3))
        with tempfile.TemporaryDirectory() as temporary:
            output = run_r(Path(temporary), data, sets, [], runs)
        print("Reference: " + ", ".join(output["versions"]))
        benchmark(data, sets, runs, output["seconds"])
        return 0
    small = arguments.write_fixtures
    data, sets, cases, cube = inputs(small)
    with tempfile.TemporaryDirectory() as temporary:
        output = run_r(Path(temporary), data, sets, cases, [])
    print("Reference: " + ", ".join(output["versions"]))
    table = []
    for case in cases:
        reference = output["cases"][case["name"]]
        result = run_cytepool(case, data, sets, cube)
        check_case(case, reference, result, data, table)
        if case["name"] == "gsva, Gaussian kernel":
            check_figure(reference, result, table)
    failures = 0
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for check, source, metric, value, tolerance in table:
        if isinstance(tolerance, str):
            result = "deliberate difference" if tolerance == "deliberate" else tolerance
            shown = "n/a"
        else:
            result = "pass" if value <= tolerance else "FAIL"
            shown = f"{tolerance:g}"
        failures += result == "FAIL"
        print(f"| {check} | {source} | {metric} | {value:.1e} | {shown} | {result} |")
    print(f"{len(table)} checks, {failures} failed")
    if small:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(data, sets, cases, output)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
