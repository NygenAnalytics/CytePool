"""Check variance partitioning against variancePartition's fitExtractVarPartModel.

Compares ``variance_fractions`` (cytepool.stats.variance_fractions) and
``variance_partitioning`` with variancePartition's ``fitExtractVarPartModel`` and
``calcVarPart``, and ``plot_variance_fractions`` and ``plot_feature_variance_fractions``
(cytepool.plotting) with its ``plotVarPart`` and ``plotPercentBars``, on data
simulated from fixed seeds (20261101 to 20261108), 60 genes each:

- ``one_random``: ``~ (1 | donor)``, 10 donors x 3 samples, a fifth of the genes
  without a donor effect;
- ``several_random``: ``~ age + rin + (1 | donor) + (1 | batch) + (1 | condition)``,
  8 donors x 4 samples over 3 batches and 2 conditions, the age of each donor and
  the RNA integrity (``rin``) of each sample, with precision weights for every
  observation; also fitted by REML (``reml``);
- ``unbalanced``: ``~ age + (1 | donor)``, donors with 1 to 4 samples, with sample
  weights;
- ``nested``: ``~ (1 | donor/tissue)``, 6 donors x 2 tissues x 2 replicates;
- ``linear``: ``~ age + rin + batch + condition``, without random effects, on the
  values and weights of ``several_random`` (variancePartition fits ``lm``);
- ``voom``: negative binomial counts of 80 genes, with the precision weights of
  ``voomWithDreamWeights(span = 0.5)`` for ``~ age + (1 | donor) + (1 | condition)``;
- ``variance_partitioning`` on three cell types of a cube of 8 donors x 2
  conditions, with prior weights from the numbers of cells and samples left out for
  having fewer than 10 cells; in the third cell type every donor keeps one sample.

The optimum: variancePartition fits every gene with lme4, run here with Powell's
BOBYQA from minqa with its trust region shrunk to 1e-10, as validation/dream.py does.
The fractions and variances at the optimum are compared to the close tier (1e-6). At
CytePool's estimates, lme4 evaluated once (``maxeval = 1``) and ``calcVarPart`` give
closed forms, compared to the exact tier (1e-10), as are the fractions of the linear
models. Differences are taken relative to the gene's total variance, the scale of a
fraction: lme4 leaves a variance component at the boundary at about 1e-16 of the
total, where CytePool holds it at zero.

The figures are drawn by R from CytePool's fractions of ``several_random``, after
``sortCols``: the order of the variables, the boxes and outliers of ``plotVarPart``
and the segments of ``plotPercentBars`` are compared to the exact tier, in percent
over 100. The violins are compared to 1e-3 of their widest point: ggplot2 takes R's
``density()``, which bins the values on a grid of 512 points and convolves them by
the fast Fourier transform (its help page), and so differs from the kernel density
estimate that CytePool draws, here by up to 6e-4 of its maximum.

Deliberate differences, printed as such: a gene that the donor effects reproduce,
REML without random effects, and a cell type with one sample per donor. Designs that
both refuse are checked too: a factor among the fixed effects of a mixed model, and a
formula without an intercept.

Requirements:
    Python 3.13 with numpy, numba and matplotlib, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 with variancePartition 1.36.2, lme4 1.1-x, minqa, ggplot2 and jsonlite
    (``RSCRIPT``, defaulting to ``Rscript``); R runs with 8 GB of address space at
    most.

Usage:
    PYTHONPATH=. python validation/variance_partitioning.py
    PYTHONPATH=. python validation/variance_partitioning.py --write-fixtures
"""

import argparse
import json
import os
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from cytepool import Cube, Design, plotting, variance_partitioning
from cytepool.normalization import filter_by_expression, library_sizes, tmm_factors
from cytepool.stats.dream import voom_with_dream_weights
from cytepool.stats.variance_fractions import variance_fractions

FIXTURES = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "variance_partitioning"
)
EXACT, CLOSE, VIOLIN = 1e-10, 1e-6, 1e-3
MEMORY = 8_000_000_000  # bytes of address space for R
N_GIVEN = 20  # genes evaluated at CytePool's estimates, and kept in the fixtures
SCALED = "max diff. / total variance"
PERCENT = "max diff. / 100"
BAR_GENES = ["g3", "g1", "g7", "g12", "g20"]

R_SCRIPT = r"""
suppressPackageStartupMessages({library(variancePartition); library(lme4); library(ggplot2)})
# lme4 1.1-38 warns once per session that these functions moved to reformulas, and
# variancePartition turns the warning of its first fit into an error.
invisible(suppressWarnings(list(findbars(~ (1 | b)), nobars(~ a + (1 | b)), subbars(~ (1 | b)))))
args <- commandArgs(trailingOnly = TRUE)
cases <- jsonlite::read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
tight_bobyqa <- function(par, fn, lower, upper, control) {
  res <- minqa::bobyqa(par, fn, lower = lower, upper = upper,
    control = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 1e5))
  list(par = res$par, fval = res$fval, conv = 0, message = res$msg)
}
tight <- lmerControl(optimizer = "tight_bobyqa", calc.derivs = FALSE, check.conv.singular = "ignore")
given <- lmerControl(optimizer = "nloptwrap", calc.derivs = FALSE, check.conv.singular = "ignore",
  restart_edge = FALSE, boundary.tol = 0, optCtrl = list(maxeval = 1))
table_of <- function(m) list(names = rownames(m), columns = colnames(m), values = unname(m))
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("variancePartition", "lme4", "minqa", "ggplot2"), function(p) as.character(packageVersion(p))))))
for (name in names(cases)) {
  case <- cases[[name]]
  data <- as.data.frame(case$samples, stringsAsFactors = TRUE)
  rownames(data) <- paste0("s", seq_len(nrow(data)))
  form <- as.formula(case$formula)
  reml <- isTRUE(case$reml)
  r <- list()
  r$error <- tryCatch({
    if (!is.null(case$counts)) {  # voom's weights, as variancePartition computes them
      counts <- case$counts
      dimnames(counts) <- list(paste0("g", seq_len(nrow(counts))), rownames(data))
      expr <- quiet(voomWithDreamWeights(counts, form, data, lib.size = case$lib_size,
        span = 0.5, weights = case$weights, rescaleWeightsAfter = !is.null(case$weights),
        control = tight))
    } else {
      E <- case$values
      dimnames(E) <- list(paste0("g", seq_len(nrow(E))), rownames(data))
      expr <- if (is.null(case$weights)) E else new("EList", list(E = E, weights = case$weights))
    }
    vp <- quiet(fitExtractVarPartModel(expr, form, data, REML = reml, control = tight,
      showWarnings = FALSE))
    "none"
  }, error = function(e) conditionMessage(e))
  if (r$error != "none") { out[[name]] <- r; next }
  r$fractions <- table_of(as.matrix(vp))
  variances <- quiet(fitVarPartModel(expr, form, data, REML = reml, control = tight,
    showWarnings = FALSE, fxn = function(fit) calcVarPart(fit, returnFractions = FALSE)))
  r$variances <- table_of(do.call(rbind, variances))
  if (!is.null(case$theta)) {  # lme4 at CytePool's variance parameters
    f <- as.formula(paste("yy", case$formula))
    at <- NULL
    for (i in seq_len(nrow(case$theta))) {
      d <- data; d$yy <- case$values[i, ]; d$ww <- case$given_weights[i, ] / mean(case$given_weights[i, ])
      m0 <- quiet(lmer(f, d, weights = ww, REML = reml, control = given))
      groups <- sub("\\.\\(Intercept\\)$", "", names(getME(m0, "theta")))
      m <- quiet(lmer(f, d, weights = ww, REML = reml, control = given,
        start = list(theta = case$theta[i, match(groups, case$components)])))
      cv <- calcVarPart(m, returnFractions = FALSE)
      at <- rbind(at, cv[match(case$variables, names(cv))])
    }
    r$given <- unname(at)
  }
  if (!is.null(case$figure)) {  # plotVarPart and plotPercentBars of CytePool's fractions
    M <- case$figure
    dimnames(M) <- list(paste0("g", seq_len(nrow(M))), case$variables)
    sorted <- sortCols(M)
    built <- ggplot_build(plotVarPart(sorted))
    violin <- built$data[[1]]
    violin <- violin[is.na(violin$quantile), ]
    box <- built$data[[2]]
    r$labels <- built$layout$panel_params[[1]]$x$get_labels()
    r$violin <- list(group = violin$group, y = violin$y, width = violin$violinwidth)
    r$box <- list(lower = box$lower, middle = box$middle, upper = box$upper,
      ymin = box$ymin, ymax = box$ymax, outliers = lapply(box$outliers, as.numeric))
    bars <- ggplot_build(plotPercentBars(sorted[case$bar_genes, ]))
    layer <- bars$data[[1]]
    fill <- bars$plot$scales$get_scales("fill")
    genes <- bars$layout$panel_params[[1]]$y$get_labels()
    r$bars <- list(gene = genes[layer$x], left = layer$ymin + 100,
      width = layer$ymax - layer$ymin, top = rev(genes),
      variable = fill$get_limits()[match(layer$fill, fill$map(fill$get_limits()))])
  }
  out[[name]] <- r
}
jsonlite::write_json(out, args[[2]], digits = NA, matrix = "rowmajor", auto_unbox = TRUE)
"""


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, ...]] = []
        self.failures = 0

    def check(
        self, check: str, reference: str, metric: str, value: float, tolerance: float
    ) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        result = "pass" if passed else "FAIL"
        self.rows.append(
            (check, reference, metric, f"{value:.1e}", f"{tolerance:.0e}", result)
        )

    def other(
        self, check: str, reference: str, metric: str, value: str, expected: Any = None
    ) -> None:
        """Information, or a deliberate difference that must be as expected."""
        self.failures += expected is False
        result = (
            "info"
            if expected is None
            else "deliberate difference"
            if expected
            else "FAIL"
        )
        self.rows.append((check, reference, metric, value, "n/a", result))

    def print(self) -> None:
        print("| check | reference | metric | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


def run_r(cases: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "partition.R").write_text(R_SCRIPT)
        (root / "cases.json").write_text(json.dumps(cases))
        limit = (MEMORY, MEMORY)
        command = [
            os.environ.get("RSCRIPT", "Rscript"),
            "--vanilla",
            str(root / "partition.R"),
            str(root / "cases.json"),
            str(root / "out.json"),
        ]
        subprocess.run(
            command,
            check=True,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, limit),
        )
        reference: dict[str, Any] = json.loads((root / "out.json").read_text())
        return reference


# Data.


def samples_of(rng: np.random.Generator, per_donor: list[int]) -> dict[str, list[Any]]:
    donor = np.repeat(np.arange(len(per_donor)), per_donor)
    n = donor.size
    return {
        "donor": [f"D{d}" for d in donor],
        "batch": [f"B{b}" for b in rng.integers(3, size=n)],
        "condition": ["ctrl" if i % 2 == 0 else "stim" for i in range(n)],
        "age": np.round(rng.normal(50.0, 10.0, len(per_donor))[donor], 1).tolist(),
        "rin": np.round(rng.normal(7.0, 1.0, n), 2).tolist(),
    }


def simulate_values(
    seed: int, samples: dict[str, list[Any]], n_genes: int
) -> np.ndarray:
    """Log-expression with random effects of the categorical variables, a fifth of the
    genes without each, and slopes of the covariates."""
    rng = np.random.default_rng(seed)
    n = len(samples["donor"])
    y = rng.normal(5.0, 0.5, n_genes) + rng.normal(size=(n, n_genes))
    for name, sd in (
        ("donor", 1.0),
        ("batch", 0.3),
        ("condition", 0.6),
        ("tissue", 0.8),
    ):
        if name in samples:
            codes = np.unique(samples[name], return_inverse=True)[1]
            scale = rng.uniform(0.0, sd, n_genes) * (rng.random(n_genes) > 0.2)
            y += (rng.normal(size=(codes.max() + 1, n_genes)) * scale)[codes]
    for name, sd in (("age", 0.03), ("rin", 0.3)):
        if name in samples:
            values = np.asarray(samples[name])
            y += np.outer(values - values.mean(), rng.normal(0.0, sd, n_genes))
    return y


def simulate_counts(
    seed: int, samples: dict[str, list[Any]], n_genes: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_mean = simulate_values(seed + 1, samples, n_genes) - 1.0
    depth = rng.uniform(5e5, 2e6, len(samples["donor"]))
    mean = np.exp(log_mean) * (depth / depth.mean())[:, np.newaxis]
    return rng.negative_binomial(10, 10 / (10 + mean))


def value_cases() -> dict[str, dict[str, Any]]:
    """Each case: samples, values or counts, fixed and random effects, weights."""
    rng = np.random.default_rng(20261101)
    several = samples_of(rng, [4] * 8)
    nested: dict[str, list[Any]] = {"donor": [f"D{d}" for d in np.repeat(range(6), 4)]}
    nested["tissue"] = ["brain", "brain", "liver", "liver"] * 6
    specifications = [
        ("one_random", samples_of(rng, [3] * 10), "~ 1", "(1 | donor)", None),
        (
            "several_random",
            several,
            "~ age + rin",
            "(1 | donor) + (1 | batch) + (1 | condition)",
            "observations",
        ),
        (
            "unbalanced",
            samples_of(rng, [1, 2, 3, 4, 1, 2, 3, 4, 2, 3]),
            "~ age",
            "(1 | donor)",
            "samples",
        ),
        ("nested", nested, "~ 1", "(1 | donor/tissue)", None),
    ]
    cases = {}
    for seed, (name, samples, fixed, random, weighted) in enumerate(
        specifications, start=20261102
    ):
        values = simulate_values(seed, samples, 60)
        shape = {"samples": values.shape[:1], "observations": values.shape}.get(
            weighted
        )
        weights = (
            None
            if shape is None
            else np.random.default_rng(seed).uniform(0.2, 3, shape)
        )
        cases[name] = {
            "samples": samples,
            "values": values,
            "fixed": fixed,
            "random": random,
            "weights": weights,
        }
    cases["reml"] = cases["several_random"] | {"reml": True}
    cases["linear"] = cases["several_random"] | {
        "fixed": "~ age + rin + batch + condition",
        "random": None,
    }
    voom = samples_of(rng, [4] * 8)
    cases["voom"] = {
        "samples": voom,
        "counts": simulate_counts(20261106, voom, 80),
        "fixed": "~ age",
        "random": "(1 | donor) + (1 | condition)",
        "weights": None,
    }
    return cases


def design_of(case: dict[str, Any]) -> Design:
    return Design(case["samples"], case["fixed"], random=case["random"])


def ours(case: dict[str, Any]) -> Any:
    design = design_of(case)
    method = "reml" if case.get("reml") else "ml"
    if "counts" in case:
        voom = voom_with_dream_weights(case["counts"], design, span=0.5)
        return variance_fractions(
            voom.log2_cpm, design, weights=voom.weights, method=method
        )
    return variance_fractions(
        case["values"], design, weights=case["weights"], method=method
    )


def r_names(variables: tuple[str, ...]) -> list[str]:
    return ["Residuals" if name == "residuals" else name for name in variables]


def observation_weights(case: dict[str, Any]) -> np.ndarray:
    n, g = case["values"].shape
    weights = case["weights"]
    return (
        np.ones((n, g))
        if weights is None
        else np.broadcast_to(np.reshape(weights, (n, -1)), (n, g))
    )


def to_r(case: dict[str, Any], fractions: Any) -> dict[str, Any]:
    """The case as R reads it, features x samples, with CytePool's variance
    parameters of its first genes."""
    random = "" if case["random"] is None else f" + {case['random']}"
    r: dict[str, Any] = {
        "samples": case["samples"],
        "formula": case["fixed"] + random,
        "reml": bool(case.get("reml")),
    }
    if "counts" in case:
        r["counts"] = case["counts"].T.tolist()
        return r
    r["values"] = case["values"].T.tolist()
    if case["weights"] is not None:
        r["weights"] = observation_weights(case).T.tolist()
    if case["random"] is not None and fractions is not None:
        k = len(fractions.random_effects)
        variances = fractions.variances[:N_GIVEN]
        r["theta"] = np.sqrt(variances[:, :k] / variances[:, -1:]).tolist()
        r["components"] = list(fractions.random_effects)
        r["variables"] = r_names(fractions.variables)
        r["given_weights"] = observation_weights(case)[:, :N_GIVEN].T.tolist()
    return r


def reference_table(ref: dict[str, Any], variables: tuple[str, ...], key: str) -> tuple:
    """R's genes (positions) and their values, in CytePool's order of variables."""
    table = ref[key]
    genes = [int(name[1:]) - 1 for name in table["names"]]
    order = [table["columns"].index(name) for name in r_names(variables)]
    return genes, np.asarray(table["values"], float)[:, order]


def scaled(ours_: np.ndarray, reference: np.ndarray) -> float:
    total = np.sum(reference, axis=1, keepdims=True)
    return float(np.max(np.abs(ours_ - reference) / total))


def compare_case(name: str, case: dict, fractions: Any, ref: dict, results: Results):
    genes, theirs = reference_table(ref, fractions.variables, "fractions")
    _, variances = reference_table(ref, fractions.variables, "variances")
    tier = EXACT if case["random"] is None else CLOSE
    results.check(
        f"{name}: fractions and variances of {len(genes)} genes",
        "fitExtractVarPartModel, calcVarPart",
        SCALED,
        max(
            scaled(fractions.fractions[genes], theirs),
            scaled(fractions.variances[genes], variances),
        ),
        tier,
    )
    if len(genes) != fractions.fractions.shape[0]:
        results.check(
            f"{name}: genes fitted by R",
            "fitExtractVarPartModel",
            "genes missing",
            fractions.fractions.shape[0] - len(genes),
            0,
        )
    if "given" in ref:
        results.check(
            f"{name}: variances at CytePool's estimates",
            "lme4 (maxeval = 1), calcVarPart",
            SCALED,
            scaled(fractions.variances[:N_GIVEN], np.asarray(ref["given"], float)),
            EXACT,
        )
    k = len(fractions.random_effects)
    if k:
        zeros = int(np.count_nonzero(fractions.fractions[:, :k] == 0.0))
        tiny = int(np.count_nonzero(theirs[:, :k] < 1e-10))
        results.other(
            f"{name}: random effects estimated at zero",
            "fitExtractVarPartModel",
            "CytePool (exactly zero); R (below 1e-10)",
            f"{zeros}; {tiny}",
        )


# A cube of three cell types.


def cube_case() -> tuple[Cube, Design, dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(20261107)
    samples = samples_of(rng, [2] * 8)
    samples["sample_id"] = [
        f"{d}_{c}" for d, c in zip(samples["donor"], samples["condition"], strict=True)
    ]
    seeds = rng.integers(1 << 30, size=3)
    counts = np.stack([simulate_counts(int(seed), samples, 60) for seed in seeds])
    n_cells = rng.integers(20, 300, size=(16, 3))
    n_cells[[2, 5, 9], 1] = [4, 7, 9]  # NK: three samples left out
    n_cells[:, 2] = np.where(np.arange(16) % 2 == 0, 50, 3)  # rare: one per donor
    counts[n_cells.T == 0] = 0
    cube = Cube(
        {"counts": counts},
        samples=samples,
        groups={"group_id": ["B", "NK", "rare"]},
        features={"feature_id": [f"gene{i}" for i in range(60)]},
        n_cells=n_cells,
    )
    design = Design(cube.samples, "~ age", random="(1 | donor) + (1 | condition)")
    inputs = {}
    for g, group in enumerate(["B", "NK", "rare"]):
        values = np.asarray(cube.layer("counts", g), float)
        used = (n_cells[:, g] >= 10) & (values.sum(axis=1) > 0)
        keep = filter_by_expression(values[used], design=design.matrix[used])
        y = values[used][:, keep]
        inputs[group] = {
            "used": used,
            "keep": keep,
            "counts": y.T.tolist(),
            "lib_size": (library_sizes(y) * tmm_factors(y)).tolist(),
            "samples": {
                key: np.asarray(value)[used].tolist()
                for key, value in samples.items()
                if key != "sample_id"
            },
            "weights": n_cells[used, g].astype(float).tolist(),
            "formula": "~ age + (1 | donor) + (1 | condition)",
        }
    return cube, design, inputs


def compare_cube(
    cube: Cube, design: Design, inputs: dict, refs: dict, results: Results
) -> Any:
    result = variance_partitioning(cube, design, prior_weights="n_cells", span=0.5)
    for group, given in inputs.items():
        ref = refs[f"cube_{group}"]
        status = result.summary["status"][result.groups.index(group)]
        if ref["error"] != "none":
            results.other(
                f"cube, {group}: one sample per donor",
                "voomWithDreamWeights",
                "error; CytePool's status",
                f"{ref['error'].splitlines()[0][:40]}; {status}",
                status == "random_effects_not_estimable",
            )
            continue
        table = result.table(group)
        rows = result.samples["group_id"] == group
        expected = [f"gene{i}" for i in np.flatnonzero(given["keep"])]
        differences = int(
            np.count_nonzero(result.samples["used"][rows] != given["used"])
        )
        differences += int(table["feature_id"].tolist() != expected)
        results.check(
            f"cube, {group}: samples used, features",
            "the rules of dream_voom",
            "differences",
            differences,
            0,
        )
        fractions = np.column_stack([table[name] for name in result.variables])
        genes, theirs = reference_table(ref, result.variables, "fractions")
        results.check(
            f"cube, {group}: fractions of {len(genes)} genes",
            "voomWithDreamWeights, fitExtractVarPartModel",
            SCALED,
            scaled(fractions[genes], theirs),
            CLOSE,
        )
    return result


# Figures.


def compare_figures(fractions: Any, ref: dict[str, Any], results: Results) -> None:
    ax = plotting.plot_variance_fractions(fractions, ax=Figure().subplots())
    labels = [label.get_text() for label in ax.get_xticklabels()]
    results.check(
        "violins: the variables and their order",
        "plotVarPart(sortCols())",
        "differences",
        float(labels != ref["labels"]),
        0,
    )
    children = ax.get_children()
    boxes = [a for a in children if a.get_gid() == "boxes"]
    lines = {
        gid: [a.get_ydata() for a in children if a.get_gid() == gid]
        for gid in ("medians", "whiskers")
    }
    box = ref["box"]
    drawn = {
        "lower": [b.get_y() for b in boxes],
        "upper": [b.get_y() + b.get_height() for b in boxes],
        "middle": [y[0] for y in lines["medians"]],
        "ymin": [y[1] for y in lines["whiskers"][0::2]],
        "ymax": [y[1] for y in lines["whiskers"][1::2]],
    }
    value = max(np.max(np.abs(np.asarray(drawn[key]) - box[key])) for key in drawn)
    outliers = [
        np.sort(a.get_offsets()[:, 1]) for a in children if a.get_gid() == "outliers"
    ]
    for drawn_outliers, theirs in zip(outliers, box["outliers"], strict=True):
        theirs = np.sort(np.atleast_1d(np.asarray(theirs, dtype=float)))
        if drawn_outliers.size != theirs.size:
            value = np.inf
        elif theirs.size:
            value = max(value, float(np.max(np.abs(drawn_outliers - theirs))))
    results.check(
        "violins: quartiles, whiskers and outliers",
        "plotVarPart",
        PERCENT,
        value / 100.0,
        EXACT,
    )
    violins = {a.get_label(): a for a in children if a.get_gid() == "violins"}
    group, largest = np.asarray(ref["violin"]["group"]), 0.0
    for position, label in enumerate(ref["labels"], start=1):
        vertices = violins[label].get_paths()[0].vertices
        right = vertices[vertices[:, 0] >= position]
        order = np.argsort(right[:, 1], kind="stable")
        y, half = right[order, 1], right[order, 0] - position
        rows = group == position
        at = np.interp(np.asarray(ref["violin"]["y"])[rows], y, half)
        width = 0.45 * np.asarray(ref["violin"]["width"])[rows]
        largest = max(largest, float(np.max(np.abs(at - width)) / 0.45))
    results.check(
        "violins: half-widths at R's points, over the widest",
        "plotVarPart",
        "max diff.",
        largest,
        VIOLIN,
    )

    names = [f"g{i + 1}" for i in range(fractions.fractions.shape[0])]
    ax = plotting.plot_feature_variance_fractions(
        fractions, BAR_GENES, labels=names, ax=Figure().subplots()
    )
    segments = {}
    for container in ax.containers:
        for gene, patch in zip(BAR_GENES, container, strict=True):
            segments[gene, container.get_label()] = (patch.get_x(), patch.get_width())
    bars = ref["bars"]
    found = np.array(
        [
            segments.get(key, (np.inf, np.inf))
            for key in zip(bars["gene"], bars["variable"], strict=True)
        ]
    )
    # ggplot stacks a segment without width anywhere; only the others have a place.
    placed = np.asarray(bars["width"]) > 0.0
    value = max(
        np.max(np.abs(found[placed, 0] - np.asarray(bars["left"])[placed])),
        np.max(np.abs(found[:, 1] - bars["width"])),
    )
    top = [
        label.get_text()
        for label in sorted(
            ax.get_yticklabels(), key=lambda text: -text.get_position()[1]
        )
    ]
    results.check(
        "bars: the segments of every gene and variable",
        "plotPercentBars",
        PERCENT,
        value / 100.0 if top == list(bars["top"]) else np.inf,
        EXACT,
    )


# Deliberate differences, and designs that both refuse.


def deviation_cases() -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(20261108)
    samples = samples_of(rng, [3] * 6)
    values = simulate_values(20261108, samples, 10)
    values[:, 0] = np.repeat(rng.normal(5.0, 1.0, 6), 3)  # constant within donors
    base = {"samples": samples, "values": values, "weights": None}
    return {
        "reproduced": base | {"fixed": "~ 1", "random": "(1 | donor)"},
        "factor_fixed": base | {"fixed": "~ condition", "random": "(1 | donor)"},
        "no_intercept": base | {"fixed": "~ 0 + age", "random": "(1 | donor)"},
    }


def compare_deviations(
    cases: dict, computed: dict, refs: dict, results: Results
) -> None:
    case, ref = cases["reproduced"], refs["reproduced"]
    fractions = ours(case).fractions[0]
    theirs = dict(
        zip(ref["fractions"]["names"], ref["fractions"]["values"], strict=True)
    )
    donor = (
        theirs["g1"][ref["fractions"]["columns"].index("donor")]
        if "g1" in theirs
        else None
    )
    results.other(
        "a gene constant within every donor",
        "fitExtractVarPartModel",
        "R's donor fraction; CytePool's",
        f"{donor}; {fractions[0]:.3g}",
        bool(np.all(np.isnan(fractions))),
    )
    linear, reml = computed["linear"], ours(cases["linear"] | {"reml": True})
    r_ml = refs["linear"]["fractions"]["values"]
    r_reml = refs["linear_reml"]["fractions"]["values"]
    r_change = float(np.max(np.abs(np.asarray(r_reml) - np.asarray(r_ml))))
    ours_change = float(np.max(np.abs(reml.fractions - linear.fractions)))
    results.other(
        "REML without random effects",
        "fitExtractVarPartModel (lm)",
        "largest change of a fraction from ML: R's; CytePool's",
        f"{r_change:.1e}; {ours_change:.2g}",
        r_change == 0.0 and ours_change > 0.0,
    )
    for name in ("factor_fixed", "no_intercept"):
        try:
            ours(cases[name])
            refused = False
        except ValueError:
            refused = True
        both = refs[name]["error"] != "none" and refused
        results.check(
            f"refused by both: {name.replace('_', ' ')}",
            "fitExtractVarPartModel",
            "differences",
            float(not both),
            0,
        )


def write_fixtures(
    cases: dict, computed: dict, refs: dict, cube: Cube, variables: tuple[str, ...]
) -> None:
    """The first genes of each case and R's results for them, the cube, and the
    figures' data."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name in (
        "one_random",
        "several_random",
        "unbalanced",
        "nested",
        "linear",
        "reml",
    ):
        case, ref, fractions = cases[name], refs[name], computed[name]
        genes, theirs = reference_table(ref, fractions.variables, "fractions")
        _, variances = reference_table(ref, fractions.variables, "variances")
        first = [i for i, gene in enumerate(genes) if gene < N_GIVEN]
        arrays = {
            "values": case["values"][:, :N_GIVEN],
            "genes": np.array(genes)[first],
            "fractions": theirs[first],
            "variances": variances[first],
            "variables": np.array(fractions.variables),
        }
        weights = case["weights"]
        if weights is not None:  # sample weights, or those of the genes kept
            arrays["weights"] = weights if weights.ndim == 1 else weights[:, :N_GIVEN]
        if "given" in ref:
            arrays["given"] = np.asarray(ref["given"], float)
        np.savez_compressed(FIXTURES / f"{name}.npz", **arrays)
        text = {
            "samples": case["samples"],
            "fixed": case["fixed"],
            "random": case["random"],
            "method": "reml" if case.get("reml") else "ml",
        }
        (FIXTURES / f"{name}.json").write_text(json.dumps(text) + "\n")
    arrays = {"counts": np.asarray(cube.layer("counts")), "n_cells": cube.n_cells}
    for group in ("B", "NK"):
        genes, theirs = reference_table(refs[f"cube_{group}"], variables, "fractions")
        arrays[f"{group} genes"], arrays[f"{group} fractions"] = np.array(genes), theirs
    np.savez_compressed(FIXTURES / "cube.npz", **arrays)
    samples = {key: cube.samples[key].tolist() for key in cube.samples.columns}
    (FIXTURES / "cube.json").write_text(json.dumps(samples) + "\n")
    ref = refs["several_random"]
    bars = ref["bars"]
    np.savez_compressed(
        FIXTURES / "figures.npz",
        fractions=computed["several_random"].fractions,
        variables=np.array(computed["several_random"].variables),
        labels=np.array(ref["labels"]),
        violin_group=np.asarray(ref["violin"]["group"]),
        violin_y=np.asarray(ref["violin"]["y"]),
        violin_width=np.asarray(ref["violin"]["width"]),
        **{
            f"box_{key}": np.asarray(ref["box"][key], float)
            for key in ("lower", "middle", "upper", "ymin", "ymax")
        },
        outliers=np.concatenate(
            [np.atleast_1d(o) for o in ref["box"]["outliers"]] or [[]]
        ),
        outlier_counts=np.array(
            [np.atleast_1d(o).size for o in ref["box"]["outliers"]]
        ),
        bar_genes=np.array(BAR_GENES),
        bar_gene=np.array(bars["gene"]),
        bar_variable=np.array(bars["variable"]),
        bar_left=np.asarray(bars["left"], float),
        bar_width=np.asarray(bars["width"], float),
    )
    print(f"wrote the fixtures to {FIXTURES}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = value_cases()
    computed = {name: ours(case) for name, case in cases.items()}
    to_send = {name: to_r(case, computed[name]) for name, case in cases.items()}
    several = computed["several_random"]
    to_send["several_random"] |= {
        "figure": several.fractions.tolist(),
        "variables": r_names(several.variables),
        "bar_genes": BAR_GENES,
    }
    to_send["linear_reml"] = to_send["linear"] | {"reml": True}
    deviations = deviation_cases()
    to_send |= {name: to_r(case, None) for name, case in deviations.items()}
    cube, design, inputs = cube_case()
    to_send |= {
        f"cube_{group}": {k: v for k, v in value.items() if k not in ("used", "keep")}
        for group, value in inputs.items()
    }
    refs = run_r(to_send)
    print("Reference: " + ", ".join(f"{k} {v}" for k, v in refs["versions"].items()))
    results = Results()
    for name, case in cases.items():
        compare_case(name, case, computed[name], refs[name], results)
    result = compare_cube(cube, design, inputs, refs, results)
    compare_figures(several, refs["several_random"], results)
    compare_deviations(cases | deviations, computed, refs, results)
    results.print()
    print(f"\n{len(results.rows)} checks, {results.failures} failed")
    if arguments.write_fixtures:
        write_fixtures(cases, computed, refs, cube, result.variables)
    return 1 if results.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
