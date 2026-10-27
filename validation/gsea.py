"""Check gene set enrichment analysis (GSEA) and its figures against fgsea.

Inputs, simulated from fixed seeds: statistics of 4,000 genes like moderated
t-statistics (6% up, 5% down, a block of 60 tied at 1.25 and 30 at zero), and 90 gene
sets of 1 to 1,000 genes, a fifth of each outside the ranking, enriched for genes up,
down or both or for none, with a set of every gene (which no method tests), a set of
one gene, half of the tied block, a set that lists genes twice and two identical sets.
Each case runs ``fgsea`` (``fgseaMultilevel``, or ``fgseaSimple`` given ``nperm``) and
``gsea`` with the same options: fgsea's defaults; gseaParam 0 and 0.5; scoreType
"neg"; scoreType "pos" on statistics that are all positive; minSize 15, maxSize 500,
eps 0, sampleSize 51 and nPermSimple 2,000; permutations (nperm 2,500, in three
chunks); statistics mostly positive, so that some sets have too few random scores of
their sign and none needs the multilevel method; and one set alone.

- Exact, given fgsea's seeds: R records the seeds that fgsea draws with
  ``sample.int`` where it hands them to its C++ code (``trace`` of
  ``fgseaSimpleImpl`` and ``multilevelImpl``), and ``gsea`` draws the same through
  ``rng``. The sets tested, their sizes and leading edges must be identical, and the
  enrichment scores, normalized scores, p-values, adjusted p-values and log2 errors
  must agree to 1e-10 (relative), NaN where fgsea gives NA. fgsea tests one set alone
  with R's sampler, which cannot be replayed: that case is exact for the enrichment
  score and leading edge only.
- Calibrated, with seeds of their own (R's 1 to 20 and numpy's 1 to 20): the mean
  log10 p-value and the mean NES of each set over 20 runs must agree within 4
  standard errors of their difference, for 36 sets of 2,000 genes and for one set
  alone.
- Figures: what ``plot_enrichment`` draws (the walk, the ticks, the extremes) against
  ``plotEnrichmentData``, for three sets and gseaParam 1 and 0.5; and the bars and
  numbers of ``plot_gsea_table`` (top 5, gseaParam 0.5) against the ticks of
  ``plotEnrichmentData`` scaled as ``plotGseaTable`` scales them, and the numbers as
  ``plotGseaTable`` writes them.

Deliberate difference: in the mostly positive case, fgsea adjusts the p-values over
the sets before it leaves those with too few random scores of their sign without a
p-value; CytePool adjusts over the sets with a p-value.

With ``--write-fixtures`` the exact cases run on 300 genes and 14 sets, without the
calibrated runs, and the tests replay them.

Requirements:
    Python 3.13 with numpy, numba and matplotlib, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with fgsea, data.table, BiocParallel and jsonlite (``RSCRIPT``, by
    default ``Rscript``).

Usage:
    PYTHONPATH=. python validation/gsea.py
    PYTHONPATH=. python validation/gsea.py --write-fixtures
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib.figure
import numpy as np

from cytepool import gsea, plotting

FIXTURES = Path(__file__).resolve().parents[1] / "tests/data/gsea"
EXACT = 1e-10
CALIBRATED = 4.0  # standard errors of the difference of two means of 20 runs
RUNS = 20
BAR = 0.45  # the half-height of a row of plot_gsea_table
# fgsea's arguments and gsea's.
OPTIONS = {
    "gseaParam": "exponent", "scoreType": "score_type", "minSize": "min_size",
    "maxSize": "max_size", "eps": "eps", "sampleSize": "sample_size",
    "nPermSimple": "n_permutations", "nperm": "n_permutations",
}  # fmt: skip

R_SCRIPT = r"""
suppressPackageStartupMessages({
  library(fgsea); library(data.table); library(jsonlite); library(BiocParallel)
})
register(SerialParam())
root <- commandArgs(trailingOnly = TRUE)[[1]]
input <- fromJSON(file.path(root, "input.json"), simplifyVector = FALSE)
# The seeds fgsea draws, recorded where it hands them to its C++ code.
seen <- new.env()
invisible(trace("fgseaSimpleImpl", where = asNamespace("fgsea"), print = FALSE,
                tracer = quote(assign("simple", seeds, envir = seen))))
invisible(trace("multilevelImpl", where = asNamespace("fgsea"), print = FALSE,
                tracer = quote(assign("multilevel", seed, envir = seen))))
statistics <- function(name) setNames(unlist(input$statistics[[name]]),
                                      unlist(input$genes))
sets <- function(name) lapply(input$sets[[name]], unlist)
run <- function(case, seed) {
  set.seed(seed)
  suppressWarnings(do.call(fgsea, c(list(pathways = sets(case$sets),
    stats = statistics(case$statistics)), case$options)))
}
out <- list(cases = list(), calibrated = list(), figures = list(), bars = list())
for (case in input$cases) {
  rm(list = ls(seen), envir = seen)
  res <- run(case, case$seed)
  out$cases[[case$name]] <- list(
    pathway = res$pathway, pval = res$pval, padj = res$padj, ES = res$ES,
    NES = res$NES, size = res$size, log2err = res$log2err,
    leadingEdge = lapply(res$leadingEdge, as.character),
    simple = seen$simple, multilevel = seen$multilevel)
}
for (case in input$calibrated) {
  runs <- lapply(seq_len(input$runs), function(s) run(case, s))
  out$calibrated[[case$name]] <- list(pathway = runs[[1]]$pathway,
    pval = lapply(runs, function(r) r$pval), NES = lapply(runs, function(r) r$NES))
}
stats <- statistics("main")
for (figure in input$figures) {
  data <- plotEnrichmentData(sets("main")[[figure$set]], stats,
                             gseaParam = figure$gseaParam)
  out$figures[[length(out$figures) + 1]] <- list(
    set = figure$set, gseaParam = figure$gseaParam, rank = data$curve$rank,
    ES = data$curve$ES, ticks = data$ticks$rank, posES = data$posES,
    negES = data$negES)
}
# plotGseaTable's bars, plotEnrichmentData's ticks over the largest absolute
# statistic, and its numbers as its labels write them: "%.2f" and x.y 10^a.
res <- out$cases[[input$table_case]]
for (name in res$pathway) {
  data <- plotEnrichmentData(sets("main")[[name]], stats, gseaParam = 0.5)
  out$bars[[name]] <- list(rank = data$ticks$rank,
                           height = data$ticks$stat / data$maxAbsStat)
}
power <- function(v) { a <- floor(log10(v)); c(sprintf("%.1f", v / 10^a), a) }
# fgseaMultilevel's choice, for each count k of random scores beyond the ES, of the
# multilevel method: when its error is below that of the permutation p-value.
multilevel <- function(n, size) {
  k <- 0:n; crude <- log2((k + 1) / (n + 1))
  left <- log2(qbeta(0.025, k, n - k + 1)); right <- log2(qbeta(0.975, k + 1, n - k))
  multilevelError((k + 1) / (n + 1), size) < 0.5 * pmax(crude - left, right - crude)
}
out$multilevel <- list(n1000_size101 = multilevel(1000, 101),
                       n2000_size51 = multilevel(2000, 51))
out$numbers <- list(pathway = res$pathway, NES = sprintf("%.2f", res$NES),
                    pval = lapply(res$pval, power), padj = lapply(res$padj, power))
out$versions <- c(paste("R", getRversion()), paste("fgsea", packageVersion("fgsea")))
write_json(out, file.path(root, "output.json"), digits = I(17), na = "null",
           null = "null")
"""


class Replay(np.random.Generator):
    """A generator that draws the given seeds, fgsea's, as gsea draws its seeds."""

    def __init__(self, seeds):
        super().__init__(np.random.PCG64())
        self.seeds = np.asarray(seeds, dtype=np.int64)

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        return self.seeds[:size] - 1  # gsea's seeds run from 1, numpy's from 0


# Inputs.


def statistics(rng, n_genes):
    """Moderated t-like statistics with genes up and down and two blocks of ties,
    -log10 p-values (all positive) and mostly positive statistics; the genes up, down
    and tied."""
    t = rng.normal(size=n_genes)
    order = rng.permutation(n_genes)
    up, down = order[: n_genes * 6 // 100], order[n_genes * 95 // 100 :]
    t[up] += rng.normal(2.5, 0.7, up.size)
    t[down] -= rng.normal(2.5, 0.7, down.size)
    tied = order[n_genes // 3 : n_genes // 3 + n_genes * 3 // 200]
    t[tied] = 1.25
    t[order[n_genes // 2 : n_genes // 2 + n_genes * 3 // 400]] = 0.0
    p = rng.uniform(size=n_genes)
    p[up] = rng.beta(0.3, 20.0, up.size)
    mostly_positive = np.abs(t)
    mostly_positive[rng.choice(n_genes, n_genes // 40, replace=False)] *= -1.0
    values = {"main": t, "positive": -np.log10(p), "unbalanced": mostly_positive}
    return {name: v.tolist() for name, v in values.items()}, (up, down, tied)


def gene_sets(rng, genes, groups, n_sets, largest):
    """Sets enriched for genes up, down, both or none, a fifth outside the ranking,
    and the special sets."""
    up, down, tied = groups
    everyone = np.arange(len(genes))
    pools = {"none": [everyone], "up": [up, everyone], "down": [down, everyone],
             "both": [up, down]}  # fmt: skip
    sets = {}
    for s, size in enumerate(np.exp(rng.uniform(0.7, np.log(largest), n_sets))):
        kind = str(rng.choice(list(pools), p=[0.55, 0.2, 0.15, 0.1]))
        drawn = [rng.choice(pool, int(size) // 2 + 1) for pool in pools[kind]]
        members = [genes[i] for i in np.concatenate(drawn)]
        members += [f"OUTSIDE{i}" for i in rng.choice(9999, int(size) // 5 + 1)]
        sets[f"set_{s:02d}_{kind}"] = list(dict.fromkeys(members))
    sets["one_gene"] = [genes[up[0]]]
    sets["tied_block"] = [genes[i] for i in tied[::2]]
    sets["listed_twice"] = [genes[i] for i in up[:15]] * 2
    sets["identical_a"] = sets["identical_b"] = [genes[i] for i in down[:25]]
    return sets


def inputs(small):
    rng = np.random.default_rng(20261026)
    n_genes = 300 if small else 4000
    genes = [f"G{i:05d}" for i in range(n_genes)]
    values, groups = statistics(rng, n_genes)
    sets = gene_sets(rng, genes, groups, 8 if small else 84, 60 if small else 1000)
    sets["every_gene"] = list(genes)
    # For the mostly positive statistics: sets of genes evenly spread along their
    # ranking, whose permutation p-values are large; a set of a few genes near its top
    # and others spread along it, whose p-value, about 0.05, is moderate but above
    # those that go to the multilevel method; and sets of 25 genes from the bottom of
    # the ranking, whose negative scores only a few random sets of their size reach.
    ranked = np.argsort(-np.asarray(values["unbalanced"]), kind="stable")
    null = {f"spread_{s}": [genes[i] for i in ranked[s :: n_genes // 25]]
            for s in range(6)}  # fmt: skip
    near_top, step = 6 if small else 5, n_genes // 100
    tilted = [*ranked[: near_top * step : step],
              *ranked[near_top * step :: (n_genes - near_top * step) // (30 - near_top)]]  # fmt: skip
    null["tilted"] = [genes[i] for i in tilted[:30]]
    bottom = ranked[-75:]
    null |= {f"bottom_{s}": [genes[i] for i in bottom[s::3]] for s in range(3)}
    strongest = max(
        (name for name in sets if name.endswith("_up")), key=lambda s: len(sets[s])
    )
    cases = [
        ("defaults", "main", "main", {}),
        ("gseaParam 0", "main", "main", {"gseaParam": 0}),
        ("gseaParam 0.5", "main", "main", {"gseaParam": 0.5}),
        ("scoreType neg", "main", "main", {"scoreType": "neg"}),
        ("scoreType pos, all positive", "positive", "main", {"scoreType": "pos"}),
        ("sizes 15 to 500, eps 0, sampleSize 51, nPermSimple 2000", "main", "main",
         {"minSize": 15, "maxSize": 500, "eps": 0, "sampleSize": 51,
          "nPermSimple": 2000}),
        ("permutations, nperm 2500", "main", "main", {"nperm": 2500}),
        ("mostly positive statistics", "unbalanced", "null", {}),
        ("one set", "main", "single", {}),
    ]  # fmt: skip
    data = {
        "genes": genes, "statistics": values, "runs": RUNS, "table_case": "defaults",
        "sets": {"main": sets, "null": null, "single": {strongest: sets[strongest]}},
        "cases": [{"name": name, "statistics": s, "sets": g, "options": o,
                   "seed": 11 + i} for i, (name, s, g, o) in enumerate(cases)],
        "figures": [{"set": s, "gseaParam": g} for s in
                    (strongest, "identical_a", "tied_block") for g in (1, 0.5)],
        "calibrated": [],
    }  # fmt: skip
    if not small:  # 36 sets of up to 300 genes, and one set alone, seeds of their own
        data["sets"]["calibrated"] = gene_sets(rng, genes, groups, 31, 300)
        data["calibrated"] = [
            {"name": "36 sets", "statistics": "main", "sets": "calibrated",
             "options": {}},
            {"name": "one set alone", "statistics": "main", "sets": "single",
             "options": {}},
        ]  # fmt: skip
    return data


def run_r(root, data):
    (root / "input.json").write_text(json.dumps(data))
    (root / "compare.R").write_text(R_SCRIPT)
    rscript = os.environ.get("RSCRIPT", "Rscript")
    subprocess.run(
        [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
    )
    return json.loads((root / "output.json").read_text())


# Comparisons.


def numbers(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def relative(ours, reference):
    """Largest difference relative to max(|reference|, 1e-300); NaN must match."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape or not np.array_equal(
        np.isnan(ours), np.isnan(reference)
    ):
        return np.inf
    keep = ~np.isnan(reference)
    scale = np.maximum(np.abs(reference[keep]), 1e-300)
    return float(np.max(np.abs(ours[keep] - reference[keep]) / scale, initial=0.0))


def run_gsea(data, case, rng):
    values = data["statistics"][case["statistics"]]
    stats = dict(zip(data["genes"], values, strict=True))
    options = {OPTIONS[key]: value for key, value in case["options"].items()}
    if "nperm" in case["options"]:
        options["method"] = "permutation"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return gsea(stats, data["sets"][case["sets"]], rng=rng, **options)


def replayed(data, case, reference):
    """gsea's table, given the seeds that fgsea drew."""
    seeds = list(reference["simple"])
    if "nperm" not in case["options"]:  # the permutations' seed and the multilevel's
        seeds.append((reference["multilevel"] or [1])[0])
    return run_gsea(data, case, Replay(seeds))


def check_case(data, case, reference, table):
    ours = replayed(data, case, reference)
    row = {name: i for i, name in enumerate(ours["gene_set"].tolist())}
    names = reference["pathway"]
    at = [row.get(name, 0) for name in names]
    edges = [e.split(", ") if e else [] for e in ours["leading_edge"].tolist()]
    same = (
        sorted(row) == sorted(names)
        and ours["n_genes"][at].tolist() == reference["size"]
        and all(edges[i] == reference["leadingEdge"][j] for j, i in enumerate(at))
    )
    label = f"{case['name']}: {len(names)} set{'s' if len(names) > 1 else ''}"
    value = relative(ours["enrichment_score"][at], reference["ES"])
    table.append((f"{label}: sets tested, sizes, leading edges and enrichment scores",
                  "fgsea", "max relative difference", value if same else np.inf,
                  EXACT))  # fmt: skip
    if case["sets"] == "single":  # fgsea's sampler for one set is R's
        return
    columns = [("normalized_enrichment_score", "NES"), ("p_value", "pval")]
    if "nperm" not in case["options"]:
        columns.append(("log2_error", "log2err"))
    value = max(relative(ours[c][at], numbers(reference[r])) for c, r in columns)
    adjusted = relative(ours["adjusted_p_value"][at], numbers(reference["padj"]))
    what = "NES, p-values" + (", log2 errors" if len(columns) == 3 else "")
    if case["sets"] == "null":
        table.append((f"{label}: {what}, given fgsea's seeds", "fgsea",
                      "max relative difference", value, EXACT))  # fmt: skip
        table.append((f"{label}: adjusted p-values, over the sets with a p-value",
                      "fgsea", "max relative difference", adjusted, None))  # fmt: skip
    else:
        table.append((f"{label}: {what} and adjusted p-values, given fgsea's seeds",
                      "fgsea", "max relative difference", max(value, adjusted),
                      EXACT))  # fmt: skip


def check_calibrated(data, case, reference, table):
    names = reference["pathway"]
    runs = [run_gsea(data, case, seed) for seed in range(1, RUNS + 1)]
    for column, key, label in (
        ("p_value", "pval", "mean log10 p-value"),
        ("normalized_enrichment_score", "NES", "mean NES"),
    ):
        theirs = np.array([numbers(run) for run in reference[key]])
        mine = np.array(
            [
                [run[column][run["gene_set"].tolist().index(s)] for s in names]
                for run in runs
            ]
        )
        if key == "pval":
            theirs, mine = np.log10(theirs), np.log10(mine)
        difference = np.abs(mine.mean(axis=0) - theirs.mean(axis=0))
        error = np.sqrt((mine.var(0, ddof=1) + theirs.var(0, ddof=1)) / RUNS)
        z = np.where(difference == 0.0, 0.0, difference / np.maximum(error, 1e-300))
        table.append((f"{case['name']}: {label} of {RUNS} runs, seeds of their own",
                      "fgsea", "largest |difference| / SE", float(z.max()),
                      CALIBRATED))  # fmt: skip


def check_figures(data, output, table):
    stats = dict(zip(data["genes"], data["statistics"]["main"], strict=True))
    worst, differ = 0.0, 0
    for figure in output["figures"]:
        ax = plotting.plot_enrichment(
            stats, data["sets"]["main"][figure["set"][0]],
            exponent=figure["gseaParam"][0], ax=matplotlib.figure.Figure().subplots(),
        )  # fmt: skip
        walk = next(line for line in ax.lines if line.get_gid() == "running sum")
        ticks = next(c for c in ax.collections if c.get_gid() == "set")
        extremes = [ln.get_ydata()[0] for ln in ax.lines if ln.get_gid() == "extremes"]
        differ += not np.array_equal(walk.get_xdata(), figure["rank"])
        differ += [s[0, 0] for s in ticks.get_segments()] != figure["ticks"]
        worst = max(worst, relative(walk.get_ydata(), figure["ES"]),
                    relative(extremes, figure["posES"] + figure["negES"]))  # fmt: skip
    check = f"plot_enrichment: the walk, ticks and extremes of {len(output['figures'])} plots"
    table.append((check, "plotEnrichmentData", "max relative difference",
                  worst if differ == 0 else np.inf, EXACT))  # fmt: skip

    case = next(c for c in data["cases"] if c["name"] == data["table_case"])
    result = replayed(data, case, output["cases"][case["name"]])
    ax = plotting.plot_gsea_table(
        stats, data["sets"]["main"], result, top=5, exponent=0.5,
        ax=matplotlib.figure.Figure().subplots(),
    )  # fmt: skip
    names = [label.get_text() for label in ax.get_yticklabels()]
    segments = next(c for c in ax.collections if c.get_gid() == "gene ranks")
    segments = segments.get_segments()
    worst, differ = 0.0, 0
    for row, name in enumerate(names):
        level = len(names) - 1 - row
        mine = sorted(
            (s[0, 0], (s[1, 1] - level) / BAR) for s in segments if s[0, 1] == level
        )
        bars = output["bars"][name]
        differ += [x for x, _ in mine] != bars["rank"]
        worst = max(worst, relative([h for _, h in mine], bars["height"]))
    texts = output["numbers"]
    at = [texts["pathway"].index(name) for name in names]
    written = [texts["NES"][i] for i in at]
    for key in ("pval", "padj"):
        written += [
            rf"${texts[key][i][0]} \cdot 10^{{{texts[key][i][1]}}}$" for i in at
        ]
    differ += [t.get_text() for t in ax.texts if t.get_gid() == "numbers"] != written
    check = f"plot_gsea_table: the bars and numbers of {len(names)} sets"
    table.append((check, "plotEnrichmentData, plotGseaTable", "max relative difference",
                  worst if differ == 0 else np.inf, EXACT))  # fmt: skip


# Fixtures.


def write_fixtures(data, output):
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    keep = ("pathway", "pval", "padj", "ES", "NES", "size", "log2err", "leadingEdge",
            "simple", "multilevel")  # fmt: skip
    fixture = {
        "genes": data["genes"], "statistics": data["statistics"], "sets": data["sets"],
        "cases": data["cases"], "figures": output["figures"],
        "multilevel": output["multilevel"],
        "fgsea": {name: {key: reference.get(key) for key in keep}
                  for name, reference in output["cases"].items()},
    }  # fmt: skip
    (FIXTURES / "fgsea.json").write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote the fixtures to {FIXTURES}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    small = parser.parse_args().write_fixtures
    data = inputs(small)
    table = []
    with tempfile.TemporaryDirectory() as temporary:
        output = run_r(Path(temporary), data)
    print("Reference: " + ", ".join(output["versions"]))
    for case in data["cases"]:
        check_case(data, case, output["cases"][case["name"]], table)
    for case in data["calibrated"]:
        check_calibrated(data, case, output["calibrated"][case["name"]], table)
    check_figures(data, output, table)
    failures = 0
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for check, source, metric, value, tolerance in table:
        if tolerance is None:
            result, shown = "deliberate difference", "n/a"
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
            write_fixtures(data, output)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
