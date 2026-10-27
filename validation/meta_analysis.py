"""Check fit_meta_analysis and meta_analysis against metafor.

Simulates the effects of many features in several studies, with their standard errors,
from fixed seeds (in ``main``), and meta-analyses every feature with CytePool and with
metafor's ``rma.uni(yi, sei = ..., method = ..., test = ...)``, run feature by feature
in a separate R program on the same values: the fixed-effect model (``method =
"EE"``) and the DerSimonian-Laird, REML and Paule-Mandel estimators of the
between-study variance, each with z-tests and with the Knapp-Hartung adjustment
(``test = "knha"``). Missing values are NA, which metafor leaves out. metafor runs its
iterative estimators to convergence, REML with ``control = list(threshold = 1e-13,
maxiter = 10000)`` and Paule-Mandel with ``control = list(tol = 1e-14, maxiter =
10000)``, and also with its defaults (a threshold of 1e-5 on the change of tau^2, and
the tolerance of ``uniroot``), which stop earlier: those rows are for information.

Cases, 120 features each: 3 studies without and with large heterogeneity; 30 studies
without and with heterogeneity; 2 studies; 5 studies of which one has a standard
error 1,000 times the others'; 4 studies of which one is imprecise and far from the
others, which gives some features two maxima of the restricted likelihood; 8 studies
that measured some features only (30% of the values missing, NaN in the effect, the
standard error or both, three features measured by one study and two by none);
features of 4 studies whose between-study variance is at zero although the Hedges
estimate, REML's start, is positive; and 5 studies whose standard errors are 1e-9 to
4e-9 for effects of spread 1. Each row covers one case and method, with both tests:
the largest difference over the estimate, standard error, statistic, p-value,
confidence limits, tau^2, Q, its p-value, I^2 (in percent, as metafor reports it) and
H^2, scaled by max(|reference|, 1), and for p-values by max(p, 1e-300)
max(1, |log p|); a mismatch in the number of studies, or an iteration that did not
converge, is an infinite difference. A feature whose REML estimate has a higher
restricted likelihood than metafor's, by metafor's own ``logLik`` at both, is another
maximum: it is left out of its row and counted in a row of deliberate differences,
and a lower likelihood is a failure. Where metafor's Fisher scoring stops without
converging, it runs again with ``stepadj = 0.5``, the remedy of its help page. Tiers:
exact (1e-10) for the fixed effect and DerSimonian-Laird, closed forms; close (1e-6)
for REML and Paule-Mandel.

Then three cohorts with 5, 4 and 6 donors, each sampled before and after a
stimulation, are simulated as pseudobulk counts: A has three cell types (T, B, NK),
B the same with too few NK cells to fit them and 10% fewer genes, 20 genes of its
own and its genes in shuffled order, and C two cell types and 10% fewer genes; 30% of
the changes differ between cohorts. ``limma_voom`` tests the stimulation in every
cohort. R reads the log2 fold changes and standard errors of each cohort's tables,
merges them by feature id, meta-analyses every feature that at least ``min_studies``
cohorts tested and adjusts the p-values by ``p.adjust(method = "BH")``. Each row
compares the features and every column of ``meta_analysis``
for one setting and cell type, in the tiers above. Behaviour rows: one study with the
Knapp-Hartung adjustment (metafor falls back to a z-test, as CytePool does); a
Paule-Mandel estimate above metafor's search interval; and zero and infinite standard
errors, which CytePool refuses.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the CRAN packages metafor and jsonlite. Set ``RSCRIPT`` to the
    Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/meta_analysis.py
    PYTHONPATH=. python validation/meta_analysis.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool import Cube, Design, limma_voom, meta_analysis
from cytepool.stats.meta_analysis import fit_meta_analysis

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "meta_analysis"
)
EXACT = 1e-10
CLOSE = 1e-6
TIERS = {"fixed": EXACT, "dl": EXACT, "reml": CLOSE, "pm": CLOSE}
METHODS = {"fixed": "EE", "dl": "DL", "reml": "REML", "pm": "PM"}
TESTS = {"z": "z", "knapp_hartung": "knha"}
# metafor's outputs, in order, and the attributes of MetaAnalysisFit they match.
FIELDS = (
    "estimate",
    "standard_error",
    "statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
    "tau_squared",
    "q_statistic",
    "q_p_value",
    "i_squared",
    "h_squared",
    "n_studies",
)
# The meta-analyses of the cohorts: (method, test, min_studies). The fixture holds the
# first two.
SETTINGS = (("reml", "z", 2), ("dl", "knapp_hartung", 2), ("pm", "z", 2))
SETTINGS += (("fixed", "z", 1),)

R_SCRIPT = r"""
suppressPackageStartupMessages(library(metafor))
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, path) writeBin(as.double(x), path, size = 8)
controls <- list(tight = list(REML = list(threshold = 1e-13, maxiter = 10000),
                              PM = list(tol = 1e-14, maxiter = 10000)),
                 default = list())
# rma.uni of one feature: its 12 outputs, whether REML needed stepadj = 0.5 (the remedy
# of its help page where Fisher scoring does not converge) and the log-likelihood; NA
# where it stops, and k = 0 for a feature without studies, which it refuses.
fit <- function(y, se, method, test, control = "tight") {
  if (!any(!is.na(y) & !is.na(se))) return(c(rep(NA_real_, 11), 0, 0, NA))
  ctrl <- controls[[control]][[method]]
  if (is.null(ctrl)) ctrl <- list()
  attempt <- function(ctrl) tryCatch(suppressWarnings(rma.uni(y, sei = se,
    method = method, test = test, control = ctrl)), error = function(e) NULL)
  r <- attempt(ctrl)
  retried <- 0
  if (is.null(r) && method == "REML" && control == "tight") {
    r <- attempt(c(ctrl, stepadj = 0.5))
    retried <- 1
  }
  if (is.null(r)) return(rep(NA_real_, 14))
  c(r$beta[1], r$se, r$zval, r$pval, r$ci.lb, r$ci.ub, r$tau2, r$QE, r$QEp, r$I2,
    r$H2, r$k, retried, as.numeric(logLik(r)))
}
# The restricted log-likelihood at CytePool's REML estimates.
at <- function(y, se, tau2) tryCatch(as.numeric(logLik(suppressWarnings(
  rma.uni(y, sei = se, method = "REML", tau2 = tau2)))), error = function(e) NA_real_)
for (run in readLines(file.path(root, "runs.txt"))) {
  p <- strsplit(run, "\t")[[1]]  # case, method, test, control
  dir <- file.path(root, p[1])
  shape <- as.integer(readLines(file.path(dir, "shape.txt")))  # studies, features
  y <- matrix(rd(file.path(dir, "y.bin")), shape[1])
  se <- matrix(rd(file.path(dir, "se.bin")), shape[1])
  out <- vapply(seq_len(shape[2]), function(j) fit(y[, j], se[, j], p[2], p[3], p[4]),
                numeric(14))
  wr(out, file.path(dir, paste(p[2], p[3], p[4], "bin", sep = ".")))
  if (p[2] == "REML" && p[3] == "z" && p[4] == "tight") {
    ours <- rd(file.path(dir, "reml.bin"))
    wr(vapply(seq_len(shape[2]), function(j) at(y[, j], se[, j], ours[j]), numeric(1)),
       file.path(dir, "reml_likelihood.bin"))
  }
}
for (analysis in readLines(file.path(root, "analyses.txt"))) {
  p <- strsplit(analysis, "\t")[[1]]  # directory, method, test, min_studies, cohorts
  dir <- file.path(root, p[1])
  tabs <- lapply(p[-(1:4)], function(cohort) list(
    id = readLines(file.path(dir, paste0(cohort, ".txt"))),
    values = matrix(rd(file.path(dir, paste0(cohort, ".bin"))), ncol = 2)))
  ids <- unique(unlist(lapply(tabs, function(tab) tab$id)))
  pick <- function(column) matrix(vapply(tabs, function(tab)
    tab$values[match(ids, tab$id), column], numeric(length(ids))), ncol = length(tabs))
  y <- pick(1)
  se <- pick(2)
  keep <- which(rowSums(!is.na(y)) >= as.integer(p[4]))
  out <- vapply(keep, function(i) fit(y[i, ], se[i, ], p[2], p[3]), numeric(14))
  rows <- rbind(out[1:12, , drop = FALSE], p.adjust(out[4, ], method = "BH"), out[14, ])
  if (p[2] == "REML") {
    ours <- rd(file.path(dir, "reml.bin"))[match(ids[keep], readLines(file.path(dir, "reml.txt")))]
    rows <- rbind(rows, vapply(seq_along(keep), function(i) at(y[keep[i], ], se[keep[i], ],
                                                               ours[i]), numeric(1)))
  }
  writeLines(ids[keep], file.path(dir, "result.txt"))
  wr(rows, file.path(dir, "result.bin"))
}
# Behaviour: the values, or the error, and the warnings.
probe <- function(expr) {
  said <- character(0)
  value <- tryCatch(withCallingHandlers(expr, warning = function(w) {
    said <<- c(said, conditionMessage(w)); invokeRestart("muffleWarning")}),
    error = function(e) conditionMessage(e))
  if (is.character(value)) return(list(error = value, warnings = said))
  list(values = c(value$beta[1], value$se, value$zval, value$pval, value$ci.lb,
                  value$ci.ub, value$tau2, value$QE, value$QEp, value$I2, value$H2,
                  value$k), warnings = said)
}
y <- c(0.5, 0.1, 0.9, -0.2)
far <- c(-40, 0, 0, 0, 40)
probes <- list(
  one_study_knha = probe(rma.uni(0.5, sei = 0.2, test = "knha")),
  pm_above_interval = probe(rma.uni(far, sei = rep(1, 5), method = "PM")),
  pm_interval_raised = probe(rma.uni(far, sei = rep(1, 5), method = "PM",
                                     control = list(tau2.max = 1e6, tol = 1e-14))),
  zero_se_ee = probe(rma.uni(y, sei = c(0.2, 0, 0.22, 0.45), method = "EE")),
  zero_se_dl = probe(rma.uni(y, sei = c(0.2, 0, 0.22, 0.45), method = "DL")),
  zero_se_pm = probe(rma.uni(y, sei = c(0.2, 0, 0.22, 0.45), method = "PM")),
  zero_se_reml = probe(rma.uni(y, sei = c(0.2, 0, 0.22, 0.45), method = "REML")),
  infinite_se = probe(rma.uni(y, sei = c(0.2, Inf, 0.22, 0.45))))
writeLines(jsonlite::toJSON(probes, auto_unbox = TRUE, digits = NA, na = "null"),
           file.path(root, "probes.json"))
writeLines(c(paste("R", getRversion()), paste("metafor", packageVersion("metafor")),
             paste("jsonlite", packageVersion("jsonlite"))),
           file.path(root, "versions.txt"))
"""


def make_cases(rng, n=120):
    """Effects and standard errors by case name, (n_studies, n_features) each."""

    def case(k, tau2):
        se = rng.uniform(0.1, 0.4, (k, n))
        effects = rng.normal(0.0, 1.0, n) + rng.normal(0.0, np.sqrt(tau2), (k, n))
        return effects + rng.normal(0.0, se), se

    cases = {
        "3 studies, no heterogeneity": case(3, 0.0),
        "3 studies, large heterogeneity": case(3, rng.uniform(0.5, 2.0, n)),
        "30 studies, no heterogeneity": case(30, 0.0),
        "30 studies, heterogeneity": case(30, rng.uniform(0.0, 0.2, n)),
        "2 studies": case(2, rng.uniform(0.0, 0.3, n)),
    }
    y, se = case(5, rng.uniform(0.0, 0.1, n))
    se[0] *= 1000.0
    cases["5 studies, one with a huge standard error"] = (y, se)
    y, se = case(4, rng.uniform(0.0, 0.05, n))
    y[0] += rng.normal(0.0, 3.0, n)  # an imprecise study far from the others
    se[0] *= 5.0
    cases["4 studies, one imprecise and outlying"] = (y, se)
    y, se = case(8, rng.uniform(0.0, 0.1, n))
    missing = rng.random(y.shape) < 0.3
    missing[1:, :3] = True  # features measured by the first study only
    missing[0, :3] = False
    missing[:, 3:5] = True  # and by none
    where = rng.integers(0, 3, y.shape)  # NaN in the effect, the error or both
    y[missing & (where != 1)] = np.nan
    se[missing & (where != 0)] = np.nan
    cases["8 studies, some features measured in some studies only"] = (y, se)
    cases["4 studies, tau^2 at zero from a positive Hedges start"] = at_zero(rng, n)
    # Standard errors far smaller than the spread of the effects.
    cases["5 studies, standard errors 1e-8 of the spread of the effects"] = (
        rng.normal(0.0, 1.0, (5, n)),
        rng.uniform(1e-9, 4e-9, (5, n)),
    )
    return cases


def at_zero(rng, n):
    """Features of 4 studies whose Hedges estimate is positive but whose Q is below
    k - 1, where the restricted likelihood is largest at zero."""
    found = []
    while len(found) < n:
        se = np.exp(rng.normal(-1.0, 0.8, 4))
        y = rng.normal(0.0, se)
        w = 1.0 / se**2
        q = np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2)
        if np.var(y, ddof=1) > np.mean(se**2) and q < 3.0:
            found.append((y, se))
    y, se = (np.array(values).T for values in zip(*found, strict=True))
    return y, se


def cohorts(rng, n_genes):
    """Three cohorts' cubes and designs, with partly different genes and cell types."""
    shared = [f"g{j:04d}" for j in range(n_genes)]
    own = [f"b{j:03d}" for j in range(20)]  # genes that only cohort B measured
    every = shared + own
    types = ("T", "B", "NK")
    profile = {t: dict(zip(every, rng.normal(0.0, 1.5, len(every)))) for t in types}
    changed = {t: rng.random(len(every)) < 0.2 for t in types}
    effect = {
        t: dict(zip(every, np.where(changed[t], rng.normal(0.0, 1.0, len(every)), 0.0)))
        for t in types
    }
    tenth = n_genes // 10
    plan = {
        "A": (5, shared, types),
        "B": (4, rng.permutation(shared[:-tenth] + own).tolist(), types),
        "C": (6, shared[tenth:], ("T", "B")),
    }
    built = {}
    for name, (n_donors, ids, kinds) in plan.items():
        donors = [f"{name}{d}" for d in range(1, n_donors + 1)]
        samples = {
            "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
            "donor": [d for d in donors for _ in range(2)],
            "condition": ["ctrl", "stim"] * n_donors,
        }
        counts = np.zeros((len(kinds), 2 * n_donors, len(ids)), dtype=np.uint32)
        for g, kind in enumerate(kinds):
            base = np.array([profile[kind][gene] for gene in ids])
            # The cohort's own departure from the common change: heterogeneity.
            departs = rng.random(len(ids)) < 0.3
            change = np.array([effect[kind][gene] for gene in ids])
            change = change + np.where(departs, rng.normal(0.0, 0.4, len(ids)), 0.0)
            donor = {d: rng.normal(0.0, 0.2, len(ids)) for d in donors}
            for s in range(2 * n_donors):
                log_mean = base + donor[samples["donor"][s]]
                log_mean = log_mean + change * (samples["condition"][s] == "stim")
                mean = rng.uniform(5e4, 3e5) * np.exp(log_mean) / np.exp(log_mean).sum()
                counts[g, s] = rng.negative_binomial(10.0, 10.0 / (10.0 + mean))
        n_cells = rng.integers(30, 80, (2 * n_donors, len(kinds)))
        if name == "B":
            n_cells[:, kinds.index("NK")] = 3  # too few cells: NK is not fitted
        cube = Cube(
            {"counts": counts},
            samples=samples,
            groups={"group_id": list(kinds)},
            features={"feature_id": ids},
            n_cells=n_cells,
        )
        design = Design(
            cube.samples,
            "~ donor + condition",
            levels={"condition": ["ctrl", "stim"]},
            contrasts={"stim_vs_ctrl": "conditionstim"},
        )
        built[name] = (cube, design)
    return built


def cohort_analyses(built):
    """limma_voom on every cohort and the meta-analyses of every setting, and what R
    gets: (method, test, min_studies, the cohorts' tables, the meta-analysis table)
    for every group and contrast meta-analysed."""
    results = {name: limma_voom(cube, design) for name, (cube, design) in built.items()}
    analyses = []
    for method, test, fewest in SETTINGS:
        meta = meta_analysis(results, method=method, test=test, min_studies=fewest)
        for key, table in meta.tables.items():
            tables = {
                name: r.tables[key] for name, r in results.items() if key in r.tables
            }
            analyses.append((method, test, fewest, key, tables, table))
    return analyses


def write_binary(path, values):
    path.write_bytes(np.ascontiguousarray(values, dtype="<f8").tobytes())


def run_metafor(cases, analyses):
    """Run metafor on every case and on the cohorts' tables; read its outputs back."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runs = []
        for c, (y, se) in enumerate(cases.values()):
            directory = root / f"case{c}"
            directory.mkdir()
            (directory / "shape.txt").write_text(f"{y.shape[0]}\n{y.shape[1]}\n")
            write_binary(directory / "y.bin", y.T)  # R reads columns: features
            write_binary(directory / "se.bin", se.T)
            write_binary(directory / "reml.bin", fit_meta_analysis(y, se).tau_squared)
            for method in METHODS.values():
                runs += [f"case{c}\t{method}\t{test}\ttight" for test in TESTS.values()]
            runs += [f"case{c}\t{method}\tz\tdefault" for method in ("REML", "PM")]
        lines = []
        for a, (method, test, fewest, _, tables, ours) in enumerate(analyses):
            directory = root / f"analysis{a}"
            directory.mkdir()
            for cohort, table in tables.items():
                (directory / f"{cohort}.txt").write_text(
                    "\n".join(table["feature_id"]) + "\n"
                )
                values = [table["log2_fold_change"], table["standard_error"]]
                write_binary(directory / f"{cohort}.bin", values)
            (directory / "reml.txt").write_text("\n".join(ours["feature_id"]) + "\n")
            write_binary(directory / "reml.bin", ours["tau_squared"])
            settings = [METHODS[method], TESTS[test], str(fewest), *tables]
            lines.append("\t".join([f"analysis{a}", *settings]))
        (root / "runs.txt").write_text("\n".join(runs) + "\n")
        (root / "analyses.txt").write_text("\n".join(lines) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )

        def read(path, rows):
            return np.fromfile(path, dtype="<f8").reshape(-1, rows).T

        outputs, likelihoods = {}, {}
        for c, name in enumerate(cases):
            for line in runs:
                case, method, test, control = line.split("\t")
                if case == f"case{c}":
                    path = root / case / f"{method}.{test}.{control}.bin"
                    outputs[name, method, test, control] = read(path, 14)
            likelihoods[name] = read(root / f"case{c}" / "reml_likelihood.bin", 1)[0]
        merged = []
        for a, analysis in enumerate(analyses):
            directory = root / f"analysis{a}"
            rows = 15 if analysis[0] == "reml" else 14
            ids = (directory / "result.txt").read_text().split()
            merged.append((ids, read(directory / "result.bin", rows)))
        probes = json.loads((root / "probes.json").read_text())
        versions = (root / "versions.txt").read_text().split("\n")[:3]
        return outputs, likelihoods, merged, probes, versions


def difference(ours, reference, field):
    """Largest |ours - reference| over the scale the module docstring gives; equal
    values (missing ones, infinities) differ by 0, a value and a missing one by inf."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if field.endswith("p_value"):
            scale = np.maximum(reference, 1e-300)
            scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
        else:
            scale = np.maximum(np.abs(reference), 1.0)
        scaled = np.abs(ours - reference) / scale
    same = (ours == reference) | (np.isnan(ours) & np.isnan(reference))
    scaled = np.where(same, 0.0, np.where(np.isnan(scaled), np.inf, scaled))
    return float(np.max(scaled, initial=0.0))


def other_maxima(own, at_ours):
    """Where CytePool's REML estimate has a higher restricted likelihood than
    metafor's (another maximum), whether it is ever lower, and the largest gain."""
    with np.errstate(invalid="ignore"):  # NaN for features without studies
        gain = at_ours - own
        return gain > 1e-9, bool(np.any(gain < -1e-9)), float(np.nanmax(gain))


def ours_as_metafor(fit):
    """A MetaAnalysisFit's values in metafor's order and units (I^2 in percent)."""
    values = [np.asarray(getattr(fit, field), float) for field in FIELDS]
    values[FIELDS.index("i_squared")] = values[FIELDS.index("i_squared")] * 100.0
    return values


def row(check, reference, value, tolerance):
    return (check, reference, "max scaled difference", f"{value:.1e}") + (
        (f"{tolerance:.0e}", "pass" if value <= tolerance else "FAIL")
        if tolerance
        else ("n/a", "information")
    )


def maxima_row(label, n, gain):
    return (
        f"{label}: REML, a higher maximum of the restricted likelihood than metafor's",
        "rma.uni, logLik",
        "features; largest gain in restricted log-likelihood",
        f"{n} of 120; {gain:.2g}",
        "n/a",
        "deliberate difference",
    )


def compare_cases(cases, outputs, likelihoods):
    rows = []
    for name, (y, se) in cases.items():
        own = outputs[name, "REML", "z", "tight"][13]
        higher, lower, gain = other_maxima(own, likelihoods[name])
        for method, code in METHODS.items():
            largest, label = 0.0, f"{name}: {method}, z and Knapp-Hartung"
            compared = np.ones(y.shape[1], dtype=bool)
            if method == "reml":
                compared, largest = ~higher, np.inf if lower else 0.0
                retried = int(np.nansum(outputs[name, code, "z", "tight"][12]))
                if retried:
                    label += f" (metafor needed stepadj = 0.5 for {retried})"
            for test, test_code in TESTS.items():
                fit = fit_meta_analysis(y, se, method=method, test=test)
                theirs = outputs[name, code, test_code, "tight"]
                for field, ours, reference in zip(
                    FIELDS, ours_as_metafor(fit), theirs[:12], strict=True
                ):
                    found = difference(ours[compared], reference[compared], field)
                    largest = max(largest, found)
                if not fit.converged.all():
                    largest = np.inf
            rows.append(row(label, "rma.uni", largest, TIERS[method]))
            if method == "reml" and higher.any():
                rows.append(maxima_row(name, int(np.count_nonzero(higher)), gain))
        for method, compared in (("reml", ~higher), ("pm", ~np.zeros_like(higher))):
            fit = fit_meta_analysis(y, se, method=method)
            theirs = outputs[name, METHODS[method], "z", "default"]
            pairs = zip(FIELDS, ours_as_metafor(fit), theirs[:12], strict=True)
            largest = max(difference(o[compared], r[compared], f) for f, o, r in pairs)
            label = f"{name}: {method}, z, against metafor's default control"
            rows.append(row(label, "rma.uni", largest, None))
    return rows


def compare_cohorts(analyses, merged, label):
    rows = []
    columns = ("log2_fold_change",) + FIELDS[1:] + ("adjusted_p_value",)
    for (method, test, fewest, key, _, table), (ids, theirs) in zip(
        analyses, merged, strict=True
    ):
        order = {feature: i for i, feature in enumerate(table["feature_id"])}
        largest, extra = np.inf, ""
        if sorted(order) == sorted(ids):
            position = [order[feature] for feature in ids]
            compared = np.ones(len(ids), dtype=bool)
            largest = 0.0
            if method == "reml":
                higher, lower, gain = other_maxima(theirs[13], theirs[14])
                compared, largest = ~higher, np.inf if lower else 0.0
                if higher.any():
                    extra = f", {np.count_nonzero(higher)} at a higher maximum"
            for i, column in enumerate(columns):
                values = np.asarray(table[column], float)[position]
                if column == "i_squared":
                    values = values * 100.0
                found = difference(values[compared], theirs[i][compared], column)
                largest = max(largest, found)
        check = (
            f"{label}, {key[0]}: {method}, {test}, min_studies={fewest}, "
            f"{len(ids)} features{extra}"
        )
        rows.append(row(check, "rma.uni, p.adjust", largest, TIERS[method]))
    return rows


def compare_probes(probes):
    def largest(fit, values):
        pairs = zip(FIELDS, ours_as_metafor(fit), values, strict=True)
        return max(difference(o, r, f) for f, o, r in pairs)

    one = fit_meta_analysis([0.5], [0.2], test="knapp_hartung")
    theirs = probes["one_study_knha"]
    agree = "values" in theirs and largest(one, theirs["values"]) <= EXACT
    rows = [
        ("one study, Knapp-Hartung", "rma.uni", "behaviour")
        + (
            f"metafor: the z-test, warning '{theirs['warnings']}'; CytePool: the z-test",
        )
        + ("n/a", "pass" if agree else "FAIL")
    ]
    far = fit_meta_analysis([-40.0, 0.0, 0.0, 0.0, 40.0], np.ones(5), method="pm")
    value = largest(far, probes["pm_interval_raised"]["values"])
    check = "Paule-Mandel above metafor's interval, against tau2.max = 1e6"
    rows.append(row(check, "rma.uni", value, CLOSE))
    stopped = probes["pm_above_interval"].get("error")
    rows.append(
        ("Paule-Mandel above metafor's interval, its default control", "rma.uni")
        + ("behaviour", f"metafor: {stopped}; CytePool: tau^2 {far.tau_squared:.6g}")
        + ("n/a", "FAIL" if stopped is None else "deliberate difference")
    )
    y = [0.5, 0.1, 0.9, -0.2]
    for label, se, names in (
        ("zero standard error", [0.2, 0.0, 0.22, 0.45], ("EE", "DL", "PM", "REML")),
        ("infinite standard error", [0.2, np.inf, 0.22, 0.45], ("REML",)),
    ):
        try:
            fit_meta_analysis(y, se)
            refused = False
        except ValueError:
            refused = True
        key = "zero_se_{}" if se[1] == 0.0 else "infinite_se"
        found = [probes[key.format(name.lower())] for name in names]
        said = [
            f"{name} stops ('{probe['error']}')"
            if "error" in probe
            else f"{name} gives estimates without Q, I^2 and H^2"
            for name, probe in zip(names, found, strict=True)
        ]
        computed = any("values" in probe for probe in found)
        rows.append(
            (label, "rma.uni", "behaviour")
            + (f"metafor: {'; '.join(said)}; CytePool: ValueError", "n/a")
            + (
                "FAIL"
                if not refused
                else "deliberate difference"
                if computed
                else "pass",
            )
        )
    return rows


def write_fixtures(cases, outputs, likelihoods, small, analyses, merged):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for c, (name, (y, se)) in enumerate(cases.items()):
        # The first 6 features, and 2 at most where CytePool's REML estimate is
        # another, higher maximum of the restricted likelihood than metafor's.
        own = outputs[name, "REML", "z", "tight"][13]
        higher = other_maxima(own, likelihoods[name])[0]
        chosen = np.r_[np.arange(6), np.flatnonzero(higher[6:])[:2] + 6]
        arrays[f"case{c} name"] = np.array(name)
        arrays[f"case{c} y"], arrays[f"case{c} se"] = y[:, chosen], se[:, chosen]
        arrays[f"case{c} reml higher"] = higher[chosen]
        for method, code in METHODS.items():
            for test, test_code in TESTS.items():
                values = outputs[name, code, test_code, "tight"]
                arrays[f"case{c} {method} {test}"] = values[:12, chosen]
    np.savez_compressed(FIXTURE_DIRECTORY / "metafor.npz", **arrays)
    arrays = {}
    for name, (cube, _) in small.items():
        arrays[f"{name} counts"] = cube.layer("counts")
        arrays[f"{name} n_cells"] = cube.n_cells
        for column in ("sample_id", "donor", "condition"):
            arrays[f"{name} {column}"] = np.array(cube.samples[column].tolist())
        arrays[f"{name} feature_id"] = np.array(cube.features["feature_id"].tolist())
        arrays[f"{name} group_id"] = np.array(cube.groups["group_id"].tolist())
    for (method, test, fewest, key, _, _), (ids, values) in zip(
        analyses, merged, strict=True
    ):
        if (method, test, fewest) in SETTINGS[:2]:
            prefix = f"{method} {test} {fewest} {key[0]} {key[1]}"
            arrays[f"{prefix} feature_id"] = np.array(ids)
            arrays[f"{prefix} values"] = values[:13]
            # Where CytePool's REML estimate is another, higher maximum than metafor's.
            higher = other_maxima(values[13], values[14])[0] if method == "reml" else 0
            arrays[f"{prefix} higher"] = np.zeros(len(ids), dtype=bool) | higher
    np.savez_compressed(FIXTURE_DIRECTORY / "cohorts.npz", **arrays)
    print(f"wrote metafor.npz and cohorts.npz to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(20261028))
    analyses = cohort_analyses(cohorts(np.random.default_rng(20261029), 400))
    count = len(analyses)
    if arguments.write_fixtures:
        small = cohorts(np.random.default_rng(20261030), 120)
        analyses += cohort_analyses(small)
    outputs, likelihoods, merged, probes, versions = run_metafor(cases, analyses)
    print("Reference: " + ", ".join(versions))
    table = compare_cases(cases, outputs, likelihoods)
    table += compare_cohorts(analyses[:count], merged[:count], "cohorts")
    if arguments.write_fixtures:
        table += compare_cohorts(analyses[count:], merged[count:], "fixture cohorts")
    table += compare_probes(probes)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in table:
        print("| " + " | ".join(line) + " |")
    failures = sum(line[-1] == "FAIL" for line in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        write_fixtures(
            cases, outputs, likelihoods, small, analyses[count:], merged[count:]
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
