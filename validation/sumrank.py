"""Check SumRank and held-out prediction against the authors' code, mpmath, a complete
enumeration of permutations and scikit-learn.

The authors of SumRank publish their R code under the Apache-2.0 licence
(https://github.com/nathan-nakatsuka/scRNA_Reproducibility, commit ``COMMIT``). This
script downloads ``SumRank.R``, ``CalibratePValueswithPermutations.R`` and
``MakingManhattanPlots.R`` of that commit into a directory outside the repository,
checks their SHA-256 and runs their functions in R as they are, on tables of each
study's results (``Gene``, ``p_val``, ``avg_log2FC``) that it writes. ``SumRank.R`` loads the unifed package for
``dirwin.hall``, the density of the Irwin-Hall distribution, which it evaluates at each
gene's capped sum; the R program replaces ``library`` by a function that does nothing
and ``dirwin.hall`` by the identity, so that the table SumRank writes holds the capped
sum itself. Their calibration compares -log10 of these values, which orders the genes
as -log10 of the density does (the density rises up to the cap), so its p-values are
those of the authors' code.

Checks:

1. The Irwin-Hall distribution function against mpmath at 200 digits (its alternating
   sum) on a grid from 1e-6 to m - 0.5, for 1 to 200 studies, and the Irwin-Hall
   p-values of ``sumrank`` for random statistics of 3, 10 and 30 studies.
2. SumRank statistics against the authors' SumRank: five studies of cases and controls
   (seed ``SEED``), cell types T and B, one study without B and one that did not
   measure 20 genes, each analysed by ``limma_voom``: ``differential_expression_sumrank``
   against SumRank on their tables, up and down (genes down with the log2 fold changes
   negated, as testing controls against cases gives them), summing every study and the
   proportions 0.65 and 0.5; and ``sumrank`` against SumRank on 4 studies of 60 genes
   whose p-values take 6 values.
3. Permutation p-values: ``differential_expression_sumrank`` with 40 permutations of the
   condition in cell type T, against the same permutations drawn here in the documented
   order and analysed by ``limma_voom`` with designs built from the cubes' sample
   tables, whose tables the authors' SumRank and ``CalibratePValueswithPermutations``
   take. Their sums are floating-point sums, so that equal sums can differ in their last
   digits; they are compared as the integers q of q / (2 (G - 1)), G the genes ranked.
   CytePool's p-value must be (1 + the null sums at most the gene's) / (1 + N); the
   authors' counts those strictly below and some of those equal, as rounding decides,
   without the one (a deliberate difference).
4. Permutations against a complete enumeration: three studies of 6 samples (3 cases)
   and 150 genes; each study's 20 assignments of the labels analysed, and their 8,000
   combinations give the exact permutation distribution, pooled over the genes, which
   1,000 random permutations must estimate within 4 standard errors, sqrt(p (1 - p) /
   1000) (calibrated).
   The authors' ``MakeManhattanPlot``, given CytePool's p-values of 20 permutations
   summing every study, with ``plot``, ``abline`` and ``text`` recording what it
   draws, must draw the points, the threshold (1e-5 higher) and the names of
   ``plot_sumrank``.
5. Held-out prediction on the studies of 2, summing 0.7 of the other studies (2 of 3
   or 4: the authors' SumRank stops when it sums one study, as its rowSums takes a
   matrix): every sample's score recomputed with scipy's ``rankdata`` from the counts
   of the aligned cubes and the genes chosen, the AUC by scikit-learn's
   ``roc_auc_score``, and the genes SumRank chose against the top genes of the
   authors' SumRank on the other studies.
6. Information: six studies without changes, the share of Irwin-Hall and permutation
   p-values at most 0.05 and 0.01, summing every study and half of them; with 60 genes
   changed in every study and 30 changed 8-fold in one study only, those found at an
   adjusted p-value below 0.05 by SumRank and by random-effects meta-analyses, and the
   mean held-out AUC of SumRank and of the meta-analysis with Knapp-Hartung.

Tolerances: 1e-12 relative for the Irwin-Hall distribution function and the statistics
(the authors' tables hold 15 significant digits); 0 for p-values formed from the same
counts and for the genes chosen; 1e-12 for the scores and the AUC.

Requirements:
    This script: numpy, numba, scipy, scikit-learn, mpmath and matplotlib, with
    cytepool from the source tree, and access to raw.githubusercontent.com on the first
    run (13 kB). R 4.4 with jsonlite: set RSCRIPT to its Rscript when it is not on PATH.
    R runs with its address space limited to 8 GB. The authors' code is kept in --code, else
    SUMRANK_CODE, else cytepool-validation/sumrank under XDG_CACHE_HOME or ~/.cache.

Usage, from the repository root:
    PYTHONPATH=. python validation/sumrank.py
    PYTHONPATH=. python validation/sumrank.py --write-fixtures
    PYTHONPATH=. python validation/sumrank.py --benchmark

``--benchmark`` times 10 permutations of 10 studies of 40 samples and 10,000 genes,
analysed by limma_voom and by limma_trend, instead of the checks.
"""

import argparse
import hashlib
import itertools
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings
from importlib.metadata import version
from pathlib import Path

import mpmath
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from cytepool.cube import Cube, CubeCollection
from cytepool.design import Design
from cytepool.differential_expression import limma_trend, limma_voom
from cytepool.held_out_prediction import held_out_prediction
from cytepool.meta_analysis import differential_expression_meta_analysis
from cytepool.sumrank import _irwin_hall_cdf, differential_expression_sumrank, sumrank

SEED = 20261030
COMMIT = "46392f66edcbaea2614a17d90bc2409648e63951"
SOURCE = "https://raw.githubusercontent.com/nathan-nakatsuka/scRNA_Reproducibility/"
FILES = {
    "SumRank.R": "78fa0054d8cfae971284733aa6fc66d3e185f4372dc4767c00518fa2f8dde163",
    "CalibratePValueswithPermutations.R": (
        "d4ba83d13474c41622ce5fa656f692ccad287a45556271ab44f57a139c3b14a2"
    ),
    "MakingManhattanPlots.R": (
        "c370aaeeffe30264f43fe202dfbb9e7ba3962d62ec3aea82b9c1c26c38cf96aa"
    ),
}
EXACT = 1e-12
MEMORY = 8_000_000 * 1024  # bytes of address space for R
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "data" / "sumrank"
CONTRAST = "case_vs_ctrl"
DIRECTIONS = (("up", 1.0), ("down", -1.0))
HELD_OUT = 0.7  # the proportion of the other studies of a held-out one

R_PROGRAM = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(jsonlite))
manifest <- fromJSON(args[1], simplifyVector = FALSE)
runs <- manifest$runs
# SumRank.R loads unifed for dirwin.hall, the Irwin-Hall density at the capped sum:
# library() does nothing here, and dirwin.hall returns the sum it is given.
library <- function(...) invisible(NULL)
dirwin.hall <- function(x, n) x
source(file.path(args[2], "SumRank.R"))
# The calibration's file starts with a snippet that reads tables; take the function.
lines <- readLines(file.path(args[2], "CalibratePValueswithPermutations.R"))
start <- grep("^CalibratePValueswithPermutations <- function", lines)
eval(parse(text = lines[start:length(lines)]))
for (run in runs) {
  setwd(run$directory)
  datasets <- unlist(run$datasets)
  presence <- data.frame(Dataset = datasets, G = unlist(run$presence))
  tables <- list()
  for (suffix in unlist(run$suffixes)) {
    SumRank(datasets, "G", suffix, unlist(run$genes), run$proportion, presence,
            run$directory)
    tables[[suffix]] <- read.table(paste0(
      "G_CombinedSignedNegLogPValranksNormalized_DirwinHallPValues_Top_",
      as.character(run$proportion), "_ofDatasets_", suffix, ".txt"), header = TRUE)
    writeBin(as.numeric(tables[[suffix]]$P_Val), paste0("sums_", suffix, ".bin"))
  }
  if (length(run$null) > 0) {
    comparison <- data.frame(NegLogPValue = unlist(lapply(
      unlist(run$null), function(suffix) tables[[suffix]]$NegLogPValue)))
    final <- CalibratePValueswithPermutations(unlist(run$genes), comparison,
                                              tables[["Real"]])
    writeBin(as.numeric(final$PVal), "calibrated.bin")
  }
}
# The authors' Manhattan plot of CytePool's p-values: plot, abline and text, which
# MakeManhattanPlot calls, record what they are given instead of drawing it.
source(file.path(args[2], "MakingManhattanPlots.R"))
drawn <- new.env()
plot <- function(x, y, ...) { drawn$x <- x; drawn$y <- y }
abline <- function(h, ...) drawn$h <- c(drawn$h, h)
text <- function(x, y, labels, ...) drawn$labels <- c(drawn$labels, as.character(labels))
MakeManhattanPlot(read.table(manifest$manhattan, header = TRUE),
                  OtherNegLogPValueCutoff = 1000, TopValueCutoff = 9, Desiredggtitle = "")
setwd(args[3])
writeBin(as.numeric(drawn$y), "manhattan_y.bin")
writeBin(as.numeric(drawn$h), "manhattan_h.bin")
writeLines(drawn$labels, "manhattan_labels.txt")
writeLines(paste("R", getRversion()), file.path(args[3], "versions.txt"))
"""


def fetch(directory):
    """The authors' two files, downloaded once and checked by their SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in FILES.items():
        path = directory / name
        if not path.exists():
            print(f"downloading {SOURCE}{COMMIT}/{name} to {directory}", flush=True)
            with urllib.request.urlopen(f"{SOURCE}{COMMIT}/{name}", timeout=60) as r:
                path.write_bytes(r.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            path.unlink()
            sys.exit(f"{name} of commit {COMMIT} does not have the SHA-256 {checksum}")
    return directory


def simulate(rng, sizes, n_genes, groups, changes, artefacts, lacking=(), missing=()):
    """Studies of cases and controls with negative binomial counts, analysed by
    limma_voom: log2 fold changes ``changes`` in every study and ``artefacts`` genes
    changed 8-fold in one study only; a study in ``lacking`` has no last group, one in
    ``missing`` no last 20 genes."""
    genes = [f"g{j:04d}" for j in range(n_genes)]
    base = rng.lognormal(3.0, 1.3, size=(len(groups), n_genes))
    cubes, own = {}, {}
    for s, n in enumerate(sizes):
        study = f"s{s + 1}"
        own[study] = rng.choice(np.flatnonzero(changes == 0), artefacts, replace=False)
        change = changes.copy()
        change[own[study]] = 3.0
        condition = np.array(["ctrl", "case"] * (n // 2))
        g = len(groups) - (study in lacking)
        mean = base[:g, None, :] * rng.lognormal(0.0, 0.25, size=(g, 1, n_genes))
        mean = mean * np.where(condition[None, :, None] == "case", 2.0**change, 1.0)
        counts = rng.negative_binomial(10, 10 / (10 + mean))
        kept = n_genes - 20 if study in missing else n_genes
        cubes[study] = Cube(
            {"counts": counts[:, :, :kept]},
            samples={"sample_id": [f"{study}_{i}" for i in range(n)], "condition": condition},
            groups={"group_id": list(groups[:g])},
            features={"feature_id": genes[:kept]},
            n_cells=np.full((n, g), 50),
        )  # fmt: skip
    collection = CubeCollection(cubes)
    cubes = {study: collection.aligned_cube(study) for study in collection.studies}
    results = {
        s: limma_voom(cube, design_of(cube.samples)) for s, cube in cubes.items()
    }
    return collection, results, own


def design_of(samples):
    levels = {"condition": ["ctrl", "case"]}
    return Design(samples, "~ condition", levels=levels, contrasts={CONTRAST: [0, 1]})


def tables_of(results, studies, group, sign):
    """Each study's tested genes, p-values and signed log2 fold changes, None where it
    has no table of the group."""
    found = {s: results[s].tables.get((group, CONTRAST)) for s in studies}
    return {
        s: None
        if t is None
        else (t["feature_id"].tolist(), t["p_value"], sign * t["log2_fold_change"])
        for s, t in found.items()
    }


class Runs:
    """Runs of the authors' SumRank, each in a directory of its own: ``tables`` maps a
    suffix, ``"Real"`` first, to each dataset's table, None where it lacks the cell
    type; the genes ranked are those of every dataset's table of the data."""

    def __init__(self, root):
        self.root, self.runs = root, {}

    def add(self, name, tables, proportion, null=()):
        directory = self.root / name
        directory.mkdir()
        for suffix, by_dataset in tables.items():
            for dataset, table in by_dataset.items():
                if table is None:
                    continue
                # Python floats print as the shortest text that reads back as them.
                lines = [f"{g}\t{float(p)!r}\t{float(c)!r}" for g, p, c in zip(*table)]
                path = directory / f"{dataset}_G_DifferentialExpression_{suffix}.txt"
                path.write_text("\n".join(["Gene\tp_val\tavg_log2FC", *lines]) + "\n")
        real = tables["Real"]
        common = set.intersection(*(set(t[0]) for t in real.values() if t is not None))
        self.runs[name] = {
            "directory": str(directory),
            "datasets": list(real),
            "presence": [int(t is not None) for t in real.values()],
            "genes": sorted(common),
            "proportion": proportion,
            "suffixes": list(tables),
            "null": list(null),
        }

    def execute(self, code, manhattan):
        """Run every run, and the authors' Manhattan plot of the table ``manhattan``
        (Gene, PVal and PVal_BH, the genes in alphabetical order)."""
        lines = [f"{g}\t{float(p)!r}\t{float(q)!r}" for g, p, q in zip(*manhattan)]
        path = self.root / "manhattan.tsv"
        path.write_text("\n".join(["Gene\tPVal\tPVal_BH", *lines]) + "\n")
        manifest = {"runs": list(self.runs.values()), "manhattan": str(path)}
        (self.root / "runs.json").write_text(json.dumps(manifest))
        (self.root / "sumrank.R").write_text(R_PROGRAM)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        files = [str(self.root / name) for name in ("sumrank.R", "runs.json")]
        subprocess.run(
            [rscript, "--vanilla", *files, str(code), str(self.root)],
            check=True,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY)),
        )
        return (self.root / "versions.txt").read_text().strip()

    def sums(self, name, suffix="Real"):
        return np.fromfile(Path(self.runs[name]["directory"]) / f"sums_{suffix}.bin")

    def grid(self, name, suffix="Real"):
        """A run's sums as the integers q of q / (2 (G - 1)), G the genes ranked."""
        scale = 2 * (len(self.runs[name]["genes"]) - 1)
        return np.rint(self.sums(name, suffix) * scale).astype(np.int64)

    def calibrated(self, name):
        return np.fromfile(Path(self.runs[name]["directory"]) / "calibrated.bin")


def relative(ours, reference):
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    differences = np.abs(ours - reference) / np.maximum(np.abs(reference), 1e-300)
    return float(np.max(np.where(ours == reference, 0.0, differences), initial=0.0))


def at(table, genes, column):
    """A column of a table at the given genes."""
    index = {g: i for i, g in enumerate(table["feature_id"].tolist())}
    return np.asarray(table[column])[[index[g] for g in genes]]


def irwin_hall():
    mpmath.mp.dps = 200

    def exact(x, m):
        x = mpmath.mpf(float(x))
        terms = range(int(mpmath.floor(x)) + 1)
        total = sum((-1) ** j * mpmath.binomial(m, j) * (x - j) ** m for j in terms)
        return float(total / mpmath.factorial(m))

    worst, count = 0.0, 0
    for m in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        fixed = [1e-6, 1e-3, 0.01, 0.5, 0.999999, 1.0, 1.000001, 1.5, 2.0]
        scaled = [0.1 * m, 0.3 * m, 0.45 * m, 0.5 * m - 1e-9, 0.5 * m, 0.7 * m, m - 0.5]
        grid = np.unique(np.clip(fixed + scaled, 1e-6, m - 1e-6))
        reference = np.array([exact(x, m) for x in grid])
        keep = reference > 1e-300  # below, both underflow to 0
        worst = max(worst, relative(_irwin_hall_cdf(grid[keep], m), reference[keep]))
        count += int(keep.sum())
    rows = [("Irwin-Hall F_m on a grid, 1 to 200 studies", "mpmath, 200 digits",
             f"max relative difference ({count} values)", worst, EXACT)]  # fmt: skip
    worst, count, rng = 0.0, 0, np.random.default_rng(SEED)
    for k in (3, 10, 30):
        fit = sumrank(rng.normal(size=(k, 400)))
        for direction, _ in DIRECTIONS:
            statistic = getattr(fit, f"{direction}_statistic")
            p = getattr(fit, f"{direction}_irwin_hall_p_value")
            below = statistic < k / 2
            reference = [exact(s, k) for s in statistic[below]]
            worst = max(worst, relative(p[below], reference))
            count += int(below.sum())
    rows.append(("Irwin-Hall p-values of sumrank, 3, 10 and 30 studies",
                 "mpmath, 200 digits", f"max relative difference ({count} genes)",
                 worst, EXACT))  # fmt: skip
    return rows


def permuted_tables(collection, results, n_permutations):
    """Group T's tables of the permutations that differential_expression_sumrank draws
    with the seed SEED: permutation by permutation and study by study, the condition
    permuted among the samples of the study's fit of T, then limma_voom."""
    tables = {"Real": tables_of(results, collection.studies, "T", 1.0)}
    generator = np.random.default_rng(SEED)
    for b in range(n_permutations):
        tables[f"Permutation_{b}"] = {}
        for study in collection.studies:
            cube, samples = collection.aligned_cube(study), results[study].samples
            used = samples["sample_id"][(samples["group_id"] == "T") & samples["used"]]
            ids = cube.samples["sample_id"].tolist()
            condition = dict(zip(ids, cube.samples["condition"].tolist()))
            new = dict(condition)
            for i, source in enumerate(generator.permutation(len(used)).tolist()):
                new[used[i]] = condition[used[source]]
            design = design_of({"sample_id": ids, "condition": [new[s] for s in ids]})
            t = limma_voom(cube, design, groups=["T"]).tables[("T", CONTRAST)]
            table = (t["feature_id"].tolist(), t["p_value"], t["log2_fold_change"])
            tables[f"Permutation_{b}"][study] = table
    return tables


def enumeration(rng):
    """Every assignment of the labels of three studies of 6 samples, against 1,000
    random permutations."""
    collection, results, _ = simulate(rng, [6] * 3, 150, ("T",), np.zeros(150), 0)
    statistics = []
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        ids, by_assignment = cube.samples["sample_id"].tolist(), []
        for cases in itertools.combinations(range(6), 3):
            condition = ["case" if i in cases else "ctrl" for i in range(6)]
            design = design_of({"sample_id": ids, "condition": condition})
            t = limma_voom(cube, design).tables[("T", CONTRAST)]
            p, change = np.asarray(t["p_value"]), np.asarray(t["log2_fold_change"])
            signed = np.where(change < 0, np.log10(p), -np.log10(p))
            by_assignment.append(dict(zip(t["feature_id"].tolist(), signed)))
        statistics.append(by_assignment)
    genes = sorted(set.intersection(*(set(s[0]) for s in statistics)))
    ranks = []
    for by_assignment in statistics:
        values = np.array([[a[g] for g in genes] for a in by_assignment])
        ranks.append((rankdata(-values, axis=1) - 1) / (len(genes) - 1))
    rows = []
    for proportion, summed in ((1, 3), (0.7, 2)):
        ours = differential_expression_sumrank(
            collection, results, "T", CONTRAST, proportion_of_studies=proportion,
            n_permutations=1000, permute="condition", rng=SEED + summed,
        ).table  # fmt: skip
        worst = 0.0
        for direction, sign in DIRECTIONS:
            u = [r if sign > 0 else 1 - r for r in ranks]
            every = np.broadcast_arrays(
                u[0][:, None, None], u[1][None, :, None], u[2][None, None, :]
            )
            best = np.sort(np.stack(every), axis=0)[:summed].sum(axis=0)
            null = np.sort(np.minimum(best, summed / 2).ravel())
            statistic = at(ours, genes, f"{direction}_statistic")
            exact = np.searchsorted(null, statistic, side="right") / null.size
            error = np.sqrt(exact * (1 - exact) / 1000) + 1 / null.size
            mine = at(ours, genes, f"{direction}_p_value")
            worst = max(worst, float(np.max(np.abs(mine - exact) / error)))
        rows.append((f"Permutation p-values, 1,000 random permutations, {summed} of 3 "
                     "studies summed", "the exact distribution of the 8,000 assignments",
                     "max |difference| / standard error", worst, 4.0))  # fmt: skip
    return rows


def recomputed_scores(collection, results, genes, held, method):
    """The score of every sample of a held-out study, from the counts of its aligned
    cube, scipy's ranks and UCell's formula, with the genes CytePool chose."""
    cube, samples = collection.aligned_cube(held), results[held].samples
    ids = cube.samples["sample_id"].tolist()
    features = {f: i for i, f in enumerate(cube.features["feature_id"].tolist())}
    total = np.zeros(len(ids))
    for group in cube.groups["group_id"].tolist():
        used = samples["sample_id"][(samples["group_id"] == group) & samples["used"]]
        counts = np.asarray(cube.layer("counts", group), float)
        for direction, sign in DIRECTIONS:
            chosen = genes["feature_id"][
                (genes["study"] == held)
                & (genes["method"] == method)
                & (genes["group_id"] == group)
                & (genes["direction"] == direction)
            ]
            columns = [features[g] for g in chosen.tolist() if g in features]
            if not columns:
                continue
            n, raw = len(columns), {}
            for s in used.tolist():
                ranks = rankdata(-counts[ids.index(s)])
                u = ranks[columns].sum() - n * (n + 1) / 2
                raw[s] = 1 - u / (n * counts.shape[1])
            low, high = min(raw.values()), max(raw.values())
            scaled = {
                s: (v - low) / (high - low) if high > low else 0.0
                for s, v in raw.items()
            }
            mean = np.mean(list(scaled.values()))
            total += sign * np.array([scaled.get(s, mean) for s in ids])
    return dict(zip(ids, total))


def held_out_rows(collection, results, prediction, runs):
    genes, scores, auc = prediction.genes, prediction.scores, prediction.auc
    worst_score = worst_auc = 0.0
    mismatched = compared = 0
    for held, method in itertools.product(
        collection.studies, ("sumrank", "meta_analysis")
    ):
        expected = recomputed_scores(collection, results, genes, held, method)
        mine = scores.filter((scores["study"] == held) & (scores["method"] == method))
        reference = [expected[s] for s in mine["sample_id"].tolist()]
        worst_score = max(worst_score, relative(mine["score"], reference))
        value = auc["auc"][(auc["study"] == held) & (auc["method"] == method)][0]
        reference_auc = roc_auc_score(mine["case"], mine["score"])
        worst_auc = max(worst_auc, abs(value - reference_auc))
    for name, run in runs.runs.items():
        parts = name.split("_")  # a held-out study's runs are named s1_T_up and so on
        if len(parts) != 3 or not parts[0][1:].isdigit():
            continue
        held, group, direction = parts
        q = runs.grid(name)
        cap = int(HELD_OUT * sum(run["presence"])) * (len(run["genes"]) - 1)
        keep = np.flatnonzero(q < cap)
        top = [run["genes"][i] for i in keep[np.lexsort((keep, q[keep]))][:15]]
        chosen = genes["feature_id"][
            (genes["study"] == held)
            & (genes["method"] == "sumrank")
            & (genes["group_id"] == group)
            & (genes["direction"] == direction)
        ]
        mismatched += chosen.tolist() != top
        compared += 1
    return [
        ("Held-out scores of every sample, 5 studies, 2 methods",
         "scipy's rankdata, UCell's formula", "max relative difference", worst_score, EXACT),
        ("Held-out AUC, 5 studies, 2 methods", "scikit-learn's roc_auc_score",
         "max absolute difference", worst_auc, EXACT),
        ("Genes SumRank chose for each held-out study, T and B, up and down",
         "the authors' SumRank on the other studies",
         f"lists that differ (of {compared})", float(mismatched), 0.0),
    ]  # fmt: skip


def manhattan_rows(result, root):
    """plot_sumrank against the authors' MakeManhattanPlot of the same p-values."""
    from matplotlib.figure import Figure

    from cytepool.plotting import plot_sumrank

    ax = plot_sumrank(result, n_labels=len(result.table), ax=Figure().subplots())
    heights = np.empty(len(result.table))
    for artist in ax.collections:
        offsets = artist.get_offsets()
        heights[np.rint(offsets[:, 0]).astype(int)] = offsets[:, 1]
    (line,) = [line for line in ax.lines if line.get_gid() == "threshold"]
    names = sorted(t.get_text() for t in ax.texts if t.get_gid() == "labels")
    theirs = np.fromfile(root / "manhattan_y.bin")
    # MakeManhattanPlot draws its line 1e-5 above the highest gene that does not pass.
    line_difference = abs(
        line.get_ydata()[0] + 1e-5 - np.fromfile(root / "manhattan_h.bin")[0]
    )
    same_names = names == sorted((root / "manhattan_labels.txt").read_text().split())
    return [
        ("Manhattan plot: every gene's -log10 p-value, in alphabetical order",
         "the authors' MakeManhattanPlot", "max relative difference",
         relative(heights, theirs), EXACT),
        ("Manhattan plot: the threshold line (theirs 1e-5 higher) and the genes named",
         "the authors' MakeManhattanPlot", "absolute difference; names that differ",
         line_difference + (not same_names), EXACT),
    ]  # fmt: skip


def simulation(rng):
    """Calibration without changes, and what each method finds with them."""
    rows = []
    collection, results, _ = simulate(rng, [16] * 6, 1200, ("T",), np.zeros(1200), 0)
    for proportion in (1, 0.5):
        out = differential_expression_sumrank(
            collection, results, "T", CONTRAST, proportion_of_studies=proportion,
            n_permutations=40, permute="condition", rng=SEED,
        )  # fmt: skip
        summed = out.settings["n_studies_summed"]
        for source, column in (
            ("Irwin-Hall", "irwin_hall_p_value"),
            ("permutation", "p_value"),
        ):
            p = np.concatenate([out.table[f"{d}_{column}"] for d in ("up", "down")])
            shares = (
                f"{np.mean(p <= 0.05):.4f} at 0.05, {np.mean(p <= 0.01):.4f} at 0.01"
            )
            rows.append((f"No changes, {summed} of 6 studies summed: {source} p-values",
                         "the nominal level", f"share of {p.size:,} p-values", shares,
                         "information"))  # fmt: skip
    changes = np.zeros(1200)
    changes[:30], changes[30:60] = 0.5, -0.5
    collection, results, own = simulate(rng, [16] * 6, 1200, ("T",), changes, 5)
    replicated = {f"g{j:04d}" for j in range(60)}
    single = {f"g{j:04d}" for genes in own.values() for j in genes}
    table = differential_expression_sumrank(
        collection, results, "T", CONTRAST, n_permutations=40, permute="condition",
        rng=SEED,
    ).table  # fmt: skip
    either = (table["up_adjusted_p_value"] < 0.05) | (
        table["down_adjusted_p_value"] < 0.05
    )
    found = {"SumRank, 40 permutations": set(table["feature_id"][either].tolist())}
    for test in ("z", "knapp_hartung"):
        meta = differential_expression_meta_analysis(results, test=test)
        meta_table = meta.table("T", CONTRAST)
        called = meta_table["feature_id"][meta_table["adjusted_p_value"] < 0.05]
        found[f"meta-analysis, REML, {test} test"] = set(called.tolist())
    for method, genes in found.items():
        counts = f"{len(genes & replicated)}; {len(genes & single)}"
        rows.append((f"60 genes changed in 6 studies, 30 in one: {method}",
                     "the simulated changes", "found of the 60; of the 30", counts,
                     "information"))  # fmt: skip
    prediction = held_out_prediction(
        collection, results, CONTRAST, "condition", "case", "ctrl", n_genes=30
    )
    for method in ("sumrank", "meta_analysis"):
        auc = prediction.auc["auc"][prediction.auc["method"] == method]
        text = f"{auc.mean():.3f} ({auc.min():.3f} to {auc.max():.3f})"
        rows.append((f"Held-out AUC of the 6 studies, genes chosen by {method}",
                     "cases against controls", "mean (smallest to largest)", text,
                     "information"))  # fmt: skip
    return rows


def benchmark():
    """The cost of permutations: 10 studies of 40 samples and 10,000 genes."""
    rng = np.random.default_rng(SEED)
    sizes, n_genes = [40] * 10, 10_000
    collection, results, _ = simulate(rng, sizes, n_genes, ("T",), np.zeros(n_genes), 0)
    for analysis in (limma_voom, limma_trend):
        if analysis is limma_trend:
            cubes = {s: collection.aligned_cube(s) for s in collection.studies}
            results = {
                s: limma_trend(c, design_of(c.samples)) for s, c in cubes.items()
            }
        options = {"permute": "condition", "rng": SEED}
        differential_expression_sumrank(  # compiles
            collection, results, "T", CONTRAST, n_permutations=1, **options
        )
        started = time.perf_counter()
        differential_expression_sumrank(
            collection, results, "T", CONTRAST, n_permutations=10, **options
        )
        seconds = (time.perf_counter() - started) / (10 * len(sizes))
        print(f"{analysis.__name__}: {seconds:.3f} s per study and permutation")


def main():
    # The statistics of sums of the best studies are compared without permutations.
    warnings.filterwarnings("ignore", "the Irwin-Hall p-values of a sum of the best")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--code", type=Path)
    arguments = parser.parse_args()
    if arguments.benchmark:
        return benchmark()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    default = cache / "cytepool-validation" / "sumrank"
    code = fetch(arguments.code or Path(os.environ.get("SUMRANK_CODE", default)))
    rows = irwin_hall()
    rng = np.random.default_rng(SEED)
    changes = np.zeros(300)
    changes[:15], changes[15:30] = 1.0, -1.0
    studies = ([12, 16, 10, 14, 20], 300, ("T", "B"), changes, 8, ("s5",), ("s2",))
    collection, results, _ = simulate(rng, *studies)
    # Tied p-values, with a log2 fold change of 0 where p is 1, as a t of 0 gives it.
    ties_p = rng.choice([1.0, 0.5, 0.2, 0.05, 0.01, 1e-4], size=(4, 60))
    ties_change = rng.choice([-1.0, 1.0], size=(4, 60))
    ties_change[(ties_p == 1.0) & (rng.uniform(size=ties_p.shape) < 0.5)] = 0.0
    # For the tests: the data, then 20 permutations, each 4 studies of 50 genes.
    small_p = rng.uniform(size=(21, 4, 50))
    small_sign = rng.choice([-1.0, 1.0], size=(21, 4, 50))
    small_p[0, :, :10] *= 0.01
    small_sign[0, :, :10] = 1.0
    prediction = held_out_prediction(
        collection, results, CONTRAST, "condition", "case", "ctrl", n_genes=15,
        proportion_of_studies=HELD_OUT,
    )  # fmt: skip
    proportions = (1, 0.65, 0.5)
    null = [f"Permutation_{b}" for b in range(40)]
    small_null = [f"Permutation_{b}" for b in range(20)]
    with tempfile.TemporaryDirectory() as temporary:
        runs = Runs(Path(temporary))
        permuted = permuted_tables(collection, results, len(null))
        for direction, sign in DIRECTIONS:
            for group, proportion in itertools.product(("T", "B"), proportions):
                tables = tables_of(results, collection.studies, group, sign)
                runs.add(
                    f"{group}_{proportion}_{direction}", {"Real": tables}, proportion
                )
            for proportion in (1, 0.5):
                tables = {d: (sorted_genes(60), ties_p[i], sign * ties_change[i])
                          for i, d in enumerate("abcd")}  # fmt: skip
                runs.add(f"ties_{proportion}_{direction}", {"Real": tables}, proportion)
            signed = {
                suffix: {d: (t[0], t[1], sign * t[2]) for d, t in by.items()}
                for suffix, by in permuted.items()
            }
            runs.add(f"permutations_{direction}", signed, 0.65, null)
            small = {
                suffix: {
                    d: (sorted_genes(50), small_p[b, i], sign * small_sign[b, i])
                    for i, d in enumerate("abcd")
                }
                for b, suffix in enumerate(["Real", *small_null])
            }
            runs.add(f"small_{direction}", small, 0.75, small_null)
            for held, group in itertools.product(collection.studies, ("T", "B")):
                if group in collection.aligned_cube(held).groups["group_id"].tolist():
                    others = [s for s in collection.studies if s != held]
                    tables = tables_of(results, others, group, sign)
                    runs.add(f"{held}_{group}_{direction}", {"Real": tables}, HELD_OUT)
        # The figure of every study summed, whose p-values from 20 permutations pass
        # the adjustment for some genes and are never 0 (MakeManhattanPlot jitters 0).
        drawn = differential_expression_sumrank(
            collection, results, "T", CONTRAST, n_permutations=20,
            permute="condition", rng=SEED,
        )  # fmt: skip
        table = drawn.table
        columns = ("feature_id", "up_p_value", "up_adjusted_p_value")
        references = runs.execute(code, [table[column] for column in columns])
        rows += manhattan_rows(drawn, runs.root)

        worst = 0.0
        for group, proportion in itertools.product(("T", "B"), proportions):
            ours = differential_expression_sumrank(
                collection, results, group, CONTRAST, proportion_of_studies=proportion
            ).table
            for direction, _ in DIRECTIONS:
                name = f"{group}_{proportion}_{direction}"
                mine = at(ours, runs.runs[name]["genes"], f"{direction}_statistic")
                worst = max(worst, relative(mine, runs.sums(name)))
        rows.append(("SumRank statistics of 5 studies, T and B, up and down, "
                     "proportions 1, 0.65 and 0.5", "the authors' SumRank",
                     "max relative difference", worst, EXACT))  # fmt: skip
        worst = 0.0
        signed = np.where(ties_change < 0, np.log10(ties_p), -np.log10(ties_p))
        for proportion in (1, 0.5):
            fit = sumrank(signed, proportion_of_studies=proportion)
            for direction, _ in DIRECTIONS:
                mine = getattr(fit, f"{direction}_statistic")
                worst = max(
                    worst, relative(mine, runs.sums(f"ties_{proportion}_{direction}"))
                )
        rows.append(("sumrank of 4 studies of 60 genes with tied p-values, proportions "
                     "1 and 0.5", "the authors' SumRank", "max relative difference",
                     worst, EXACT))  # fmt: skip

        ours = differential_expression_sumrank(
            collection, results, "T", CONTRAST, proportion_of_studies=0.65,
            n_permutations=len(null), permute="condition", rng=SEED,
        ).table  # fmt: skip
        differences = deliberate = 0.0
        outside = 0
        for direction, _ in DIRECTIONS:
            name = f"permutations_{direction}"
            q = runs.grid(name)
            pooled = np.sort(np.concatenate([runs.grid(name, s) for s in null]))
            at_most = np.searchsorted(pooled, q, side="right")
            below = np.searchsorted(pooled, q, side="left")
            mine = at(ours, runs.runs[name]["genes"], f"{direction}_p_value")
            expected = (1 + at_most) / (1 + pooled.size)
            differences = max(differences, float(np.max(np.abs(mine - expected))))
            theirs = runs.calibrated(name)
            count = theirs * pooled.size  # a count, but for the rounding of 1 - k / N
            wrong = (count < below - 1e-6) | (count > at_most + 1e-6)
            outside += int(np.count_nonzero(wrong))
            deliberate = max(deliberate, float(np.max(np.abs(mine - theirs))))
        rows += [
            ("Permutation p-values of 40 permutations, up and down",
             "(1 + the authors' null sums at most the gene's) / (1 + N)",
             "max absolute difference", differences, 0.0),
            ("The authors' calibration: null sums below the gene's, and some equal "
             "to it, divided by N", "CalibratePValueswithPermutations",
             "genes outside these bounds", float(outside), 0.0),
            ("The authors' calibrated p-values, against CytePool's",
             "CalibratePValueswithPermutations", "max absolute difference",
             deliberate, "deliberate difference"),
        ]  # fmt: skip
        rows += held_out_rows(collection, results, prediction, runs)
        fixtures = {"ties_p": ties_p, "ties_change": ties_change}
        fixtures |= {"small_p": small_p, "small_sign": small_sign}
        for name in runs.runs:
            if name.startswith(("ties_", "small_")):
                fixtures[f"{name}_sums"] = runs.sums(name)
            if name.startswith("small_"):
                sums = [runs.sums(name, suffix) for suffix in small_null]
                fixtures[f"{name}_null_sums"] = np.concatenate(sums)
                fixtures[f"{name}_calibrated"] = runs.calibrated(name)
    rows += enumeration(rng)
    rows += simulation(np.random.default_rng(SEED + 1))

    print(f"References: {references}, the authors' code at commit {COMMIT[:8]}, "
          f"mpmath {mpmath.__version__}, scikit-learn {version('scikit-learn')}, scipy "
          f"{version('scipy')}; CytePool with numpy {np.__version__}, numba "
          f"{version('numba')}")  # fmt: skip
    print("\n| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    failed = 0
    for check, reference, metric, value, tolerance in rows:
        if isinstance(tolerance, str):  # "information" or "deliberate difference"
            result, shown = tolerance, "n/a"
        else:
            result = "pass" if value <= tolerance else "FAIL"
            shown, failed = f"{tolerance:g}", failed + (result == "FAIL")
        text = value if isinstance(value, str) else f"{value:.3g}"
        print(f"| {check} | {reference} | {metric} | {text} | {shown} | {result} |")
    if arguments.write_fixtures:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FIXTURES / "authors.npz", **fixtures)
        print(f"wrote {FIXTURES / 'authors.npz'}")
    sys.exit(1 if failed else 0)


def sorted_genes(n):
    return [f"g{j:04d}" for j in range(n)]


if __name__ == "__main__":
    main()
