"""Check propeller against speckle's propeller, propeller.ttest and propeller.anova.

Simulates the number of cells of every sample in every cell type (Dirichlet-
multinomial; seeds in ``main``), runs CytePool's ``propeller`` on cubes that hold
them, and runs speckle on the same cells in R: ``propeller`` on one label per cell
for two groups (a t-test) and three (an F-test), and ``propeller.ttest`` and
``propeller.anova`` on ``getTransformedProps`` with design matrices from
``model.matrix``. Every case runs with the logit and the arcsine square-root
transform, and with speckle's defaults (robust moderation, no trend) unless a row
says otherwise.

Cases: two groups of 5 samples and 12 cell types, three of which change, also with
``robust = FALSE`` and ``trend = TRUE``; three groups of 4 samples (F-test); three
groups and a covariate, ``~ 0 + group + age``, with the contrasts B - A and (A + B)/2
- C, the F-test of ``group`` and the slope of age (against limma's ``eBayes`` on
speckle's transformed proportions, as speckle refuses it), and ``~ group + age``
with the contrast ``groupB``; a pairing, ``~ 0 + condition + donor`` (6 donors x 2
conditions), and ``~ donor + condition``; samples with 5 to 40 cells and cell types
absent from some samples, in a cube that also holds a cell type and a sample without
cells; two cell types, where limma does not moderate; and samples that are copies of
one another within each condition.

A row compares every column speckle reports, matched by cell type: ``BaselineProp``,
``PropMean`` (by level), ``PropRatio``, ``Tstatistic`` or ``Fstatistic``, ``P.Value``
and ``FDR``; the columns themselves must match. Proportions and ratios are compared
relative to speckle's in the exact tier (1e-10). Statistics are scaled by
max(|reference|, 1), and p-values and FDRs by max(p, 1e-300) max(1, |log p|), in the
exact tier, or the close tier (1e-6) with robust moderation of 10 cell types or more,
whose prior limma solves only to a tolerance.

Deliberate differences: speckle stops with an error for the slope of a covariate,
which CytePool tests, and reports a "ratio" for ``groupA + groupB``, which compares
nothing and has none in CytePool; speckle keeps an unused level of a factor of cell
labels, or of sample labels, where CytePool leaves out a cell type or a sample
without cells (with the arcsine transform such a sample has NaN proportions, and
limma leaves it out too); and CytePool refuses samples that are copies of one another
within each condition, where speckle tests the rounding errors.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages speckle and limma, and jsonlite.
    Set ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/composition_tests.py
    PYTHONPATH=. python validation/composition_tests.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool.composition import propeller
from cytepool.cube import Cube
from cytepool.design import Design

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "composition_tests"
)
EXACT = 1e-10
CLOSE = 1e-6
# speckle's columns and CytePool's; PropMean.<level> is mean_proportion_<level>.
KINDS = {
    "P.Value": "p",
    "FDR": "p",
    "Tstatistic": "statistic",
    "Fstatistic": "statistic",
}
COLUMNS = {
    "BaselineProp.Freq": "baseline_proportion",
    "BaselineProp": "baseline_proportion",
    "PropRatio": "proportion_ratio",
    "Tstatistic": "t",
    "Fstatistic": "f_statistic",
    "P.Value": "p_value",
    "FDR": "adjusted_p_value",
}
LEVELS = {"condition": ["ctrl", "stim"]}
TWO_GROUPS = {"ctrl_vs_stim": "conditionctrl - conditionstim"}
WRAPPER = {"kind": "propeller", "group": "condition"}  # propeller(clusters, ...)
FORMULA = "~ 0 + group + age"
TTEST = {"kind": "ttest", "formula": FORMULA}

R_SCRIPT = r"""
suppressPackageStartupMessages({library(speckle); library(limma)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
write_result <- function(df, path) {
  out <- data.frame(row = rownames(df))
  for (column in names(df)) {
    v <- df[[column]]
    out[[column]] <- if (is.numeric(v)) sprintf("%.17g", v) else as.character(v)
  }
  write.table(out, path, sep = "\t", quote = FALSE, row.names = FALSE)
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  groups <- readLines(file.path(dir, "groups.txt"))
  variables <- jsonlite::fromJSON(file.path(dir, "variables.json"),
                                  simplifyVector = FALSE)
  samples <- read.delim(file.path(dir, "samples.tsv"), colClasses = "character")
  for (v in names(variables)) samples[[v]] <- if (length(variables[[v]]) == 0)
    as.numeric(samples[[v]]) else factor(samples[[v]], levels = unlist(variables[[v]]))
  n <- nrow(samples)
  counts <- matrix(rd(file.path(dir, "n_cells.bin")), nrow = n)  # samples x groups
  # One label per cell, as speckle reads them.
  cl <- rep(rep(groups, each = n), times = as.vector(counts))
  s <- rep(rep(samples$sample_id, times = length(groups)), times = as.vector(counts))
  calls <- jsonlite::fromJSON(file.path(dir, "calls.json"), simplifyVector = FALSE)
  for (k in seq_along(calls)) {
    call <- calls[[k]]
    clusters <- if (is.null(call$cluster_levels)) cl else
      factor(cl, levels = unlist(call$cluster_levels))
    ids <- if (is.null(call$sample_levels)) s else
      factor(s, levels = unlist(call$sample_levels))
    result <- tryCatch(suppressMessages(if (call$kind == "propeller") {
      g <- samples[[call$group]][match(s, samples$sample_id)]
      propeller(clusters = clusters, sample = ids, group = g,
                transform = call$transform, robust = call$robust, trend = call$trend)
    } else {
      props <- getTransformedProps(clusters, ids, transform = call$transform)
      # The design's rows in the order of the columns of the proportions.
      rows <- samples[match(colnames(props$Counts), samples$sample_id), ]
      X <- model.matrix(as.formula(call$formula), rows)
      if (call$kind == "ttest") {
        propeller.ttest(props, X, contrasts = unlist(call$contrast),
                        robust = call$robust, trend = call$trend, sort = FALSE)
      } else if (call$kind == "anova") {
        propeller.anova(props, X, coef = unlist(call$coef),
                        robust = call$robust, trend = call$trend, sort = FALSE)
      } else {  # limma on speckle's transformed proportions
        fit <- contrasts.fit(lmFit(props$TransformedProps, X), unlist(call$contrast))
        eb <- eBayes(fit, robust = call$robust, trend = call$trend)
        data.frame(Tstatistic = eb$t[, 1], P.Value = eb$p.value[, 1],
                   FDR = p.adjust(eb$p.value[, 1], "BH"), row.names = rownames(eb$t))
      }
    }), error = function(e) conditionMessage(e))
    if (is.character(result)) {
      writeLines(result, file.path(dir, sprintf("call%d_error.txt", k - 1)))
    } else {
      write_result(result, file.path(dir, sprintf("call%d.tsv", k - 1)))
    }
  }
}
writeLines(c(paste("R", getRversion()), paste("speckle", packageVersion("speckle")),
             paste("limma", packageVersion("limma"))), file.path(root, "versions.txt"))
"""


class Case:
    """Cell numbers and samples, with the analyses that compare the two programs.

    An analysis is (label, test, design, call, outcome): the CytePool test, a
    contrast name or "F:<factor>" for the F-test of a factor; the design, when not
    the case's own; the speckle call, whose transform, robust and trend CytePool
    takes too; and "pass" or "deliberate".
    """

    def __init__(self, n_cells, samples, variables, design, groups=None):
        n_samples, n_groups = n_cells.shape
        self.groups = groups or [f"ct{g + 1:02d}" for g in range(n_groups)]
        self.cube = Cube(
            {"counts": np.zeros((n_groups, n_samples, 0), dtype=np.uint32)},
            samples=samples,
            groups={"group_id": self.groups},
            features={"feature_id": np.array([], dtype=object)},
            n_cells=n_cells,
        )
        self.samples, self.variables, self.design = samples, variables, design
        self.analyses = []

    def add(self, label, test, call, design=None, outcome="pass", transforms=None):
        for t in transforms or ("logit", "asin"):
            full = {"robust": True, "trend": False} | call | {"transform": t}
            name = f"{t}, {label}" if label else t
            self.analyses.append((name, test, design, full, outcome))

    def run(self, test, design, call):
        """CytePool's table of the test."""
        design = design or self.design
        options = {key: call[key] for key in ("transform", "robust", "trend")}
        if test.startswith("F:"):
            result = propeller(
                self.cube, design, contrasts=[], f_test=test[2:], **options
            )
            return result.f_test
        tested = design.contrasts.select(test)
        return propeller(self.cube, design, contrasts=tested, **options).table(test)


def simulate(rng, conditions, baseline, changes, cells, dispersion=40.0):
    """Cell numbers, samples x cell types, from a Dirichlet-multinomial model.

    Each sample draws its proportions from a Dirichlet distribution around the
    baseline proportions (one row, or one per sample) times the changes of its
    condition, then between ``cells[0]`` and ``cells[1] - 1`` cells from them.
    """
    baseline = np.broadcast_to(baseline, (len(conditions), np.shape(baseline)[-1]))
    rows = []
    for condition, base in zip(conditions, baseline, strict=True):
        mean = base * changes.get(condition, 1.0)
        proportions = rng.dirichlet(dispersion * mean / mean.sum())
        rows.append(rng.multinomial(rng.integers(*cells), proportions))
    return np.array(rows, dtype=np.int64)


def two_group_case(rng, n_types, n_per_group, cells, prefix):
    conditions = ["ctrl"] * n_per_group + ["stim"] * n_per_group
    stim = np.ones(n_types)
    stim[[0, n_types // 3, 2 * n_types // 3]] = [2.0, 0.5, 1.5]
    baseline = np.geomspace(0.005, 0.3, n_types)
    n = simulate(rng, conditions, baseline, {"stim": stim}, cells)
    ids = [f"{prefix}{i + 1:02d}" for i in range(len(conditions))]
    samples = {"sample_id": ids, "condition": conditions}
    design = Design(samples, "~ 0 + condition", levels=LEVELS, contrasts=TWO_GROUPS)
    return Case(n, samples, LEVELS, design)


def three_group_case(rng, n_types, groups, cells, prefix, age=False):
    changes = {"B": np.ones(n_types), "C": np.ones(n_types)}
    changes["B"][[1, n_types // 2]] = [3.0, 0.4]
    changes["C"][n_types // 3] = 2.0
    n = simulate(rng, groups, np.geomspace(0.005, 0.3, n_types), changes, cells)
    samples = {"sample_id": [f"{prefix}{i + 1:02d}" for i in range(len(groups))]}
    samples["group"] = groups
    variables = {"group": ["A", "B", "C"]}
    if not age:
        return Case(n, samples, variables, Design(samples, "~ 0 + group"))
    samples["age"] = rng.uniform(25.0, 70.0, len(groups)).round(1)
    contrasts = {"B_vs_A": "groupB - groupA", "AB_vs_C": "(groupA + groupB)/2 - groupC"}
    contrasts |= {"age": "age", "A_plus_B": "groupA + groupB"}
    design = Design(samples, FORMULA, contrasts=contrasts)
    return Case(n, samples, variables | {"age": []}, design)


def two_group_calls(case):
    """speckle's propeller, also with robust = FALSE and trend = TRUE."""
    case.add("", "ctrl_vs_stim", WRAPPER)
    for option in ({"robust": False}, {"trend": True}):
        label = ", ".join(f"{k} = {str(v).upper()}" for k, v in option.items())
        case.add(label, "ctrl_vs_stim", WRAPPER | option, transforms=["logit"])


def covariate_calls(case, transforms=None):
    """B - A, (A + B)/2 - C, the F-test of group, the slope of age and A + B."""
    weights = {
        name: case.design.contrasts[name].tolist() for name in case.design.contrasts
    }
    for name in ("B_vs_A", "AB_vs_C"):
        case.add(name, name, TTEST | {"contrast": weights[name]}, transforms=transforms)
    anova = {"kind": "anova", "formula": FORMULA, "coef": [1, 2, 3]}
    case.add("F-test of group", "F:group", anova, transforms=transforms)
    age = {"formula": FORMULA, "contrast": weights["age"]}
    case.add(
        "age, against limma", "age", {"kind": "limma"} | age, transforms=transforms
    )
    call = TTEST | {"contrast": weights["A_plus_B"]}
    case.add("A + B", "A_plus_B", call, outcome="deliberate", transforms=transforms)


def make_cases(rng):
    cases = {}
    case = cases["two groups"] = two_group_case(rng, 12, 5, (1000, 5000), "S")
    two_group_calls(case)

    groups = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    case = cases["three groups"] = three_group_case(rng, 12, groups, (800, 4000), "T")
    case.add("", "F:group", WRAPPER | {"group": "group"})

    groups = ["A", "B", "C"] * 5
    case = cases["covariate"] = three_group_case(rng, 12, groups, (1000, 4000), "U", 1)
    covariate_calls(case)
    other = Design(case.samples, "~ group + age", contrasts={"B_vs_A": "groupB"})
    weights = case.design.contrasts["B_vs_A"].tolist()
    case.add("~ group + age, groupB", "B_vs_A", TTEST | {"contrast": weights}, other)
    age = {"formula": FORMULA, "contrast": case.design.contrasts["age"].tolist()}
    case.add("age", "age", TTEST | age, outcome="deliberate")

    # A pairing: 6 donors in two conditions, each donor with its own composition.
    donors = [f"D{d}" for d in range(1, 7) for _ in range(2)]
    conditions = ["ctrl", "stim"] * 6
    stim = {"stim": np.r_[1.8, 1, 1, 1, 0.6, 1, 1, 1, 1, 1]}
    baseline = np.geomspace(0.005, 0.3, 10) * rng.lognormal(0.0, 0.3, (6, 10))
    n = simulate(rng, conditions, baseline.repeat(2, axis=0), stim, (1500, 3000))
    ids = [f"{d}_{c}" for d, c in zip(donors, conditions, strict=True)]
    samples = {"sample_id": ids, "condition": conditions, "donor": donors}
    contrast = {"stim_vs_ctrl": "conditionstim - conditionctrl"}
    formula = "~ 0 + condition + donor"
    design = Design(samples, formula, levels=LEVELS, contrasts=contrast)
    variables = LEVELS | {"donor": sorted(set(donors))}
    case = cases["pairing"] = Case(n, samples, variables, design)
    weights = design.contrasts["stim_vs_ctrl"].tolist()
    call = {"kind": "ttest", "formula": formula, "contrast": weights}
    case.add(formula, "stim_vs_ctrl", call)
    contrast = {"stim_vs_ctrl": "conditionstim"}
    paired = Design(samples, "~ donor + condition", levels=LEVELS, contrasts=contrast)
    case.add("~ donor + condition", "stim_vs_ctrl", call, paired)

    # Few cells and rare cell types, with a cell type and a sample without cells.
    conditions = ["ctrl"] * 4 + ["stim"] * 4
    baseline = np.r_[0.01, 0.02, 0.1, 0.2, 0.3, 0.37, 0.003, 0.002]
    n = simulate(rng, conditions, baseline, {}, (5, 41), dispersion=15.0)
    n = np.pad(n, ((0, 1), (0, 1)))  # the sample V_empty and the cell type "empty"
    ids = [f"V{i + 1}" for i in range(8)] + ["V_empty"]
    samples = {"sample_id": ids, "condition": [*conditions, "ctrl"]}
    groups = [f"ct{g + 1}" for g in range(8)] + ["empty"]
    design = Design(samples, "~ 0 + condition", levels=LEVELS, contrasts=TWO_GROUPS)
    case = cases["few cells"] = Case(n, samples, LEVELS, design, groups)
    case.add("", "ctrl_vs_stim", WRAPPER)
    call = {"kind": "ttest", "formula": "~ 0 + condition", "contrast": [1, -1]}
    call["sample_levels"] = ids
    label = "the sample without cells as an unused level"
    case.add(label, "ctrl_vs_stim", call, None, "deliberate", ["logit"])
    case.add(label, "ctrl_vs_stim", call, transforms=["asin"])
    label = "the cell type without cells as an unused level"
    call = WRAPPER | {"cluster_levels": groups}
    case.add(label, "ctrl_vs_stim", call, outcome="deliberate", transforms=["logit"])

    case = cases["two cell types"] = two_group_case(rng, 2, 3, (500, 2000), "W")
    case.add("", "ctrl_vs_stim", WRAPPER)

    # The samples of each condition are copies of one another.
    n = simulate(rng, ["ctrl", "stim"], np.geomspace(0.01, 0.3, 6), {}, (500, 1000))
    conditions = ["ctrl"] * 3 + ["stim"] * 3
    samples = {"sample_id": [f"X{i + 1}" for i in range(6)], "condition": conditions}
    design = Design(samples, "~ 0 + condition", levels=LEVELS, contrasts=TWO_GROUPS)
    case = cases["copies"] = Case(n.repeat(3, axis=0), samples, LEVELS, design)
    case.add("", "ctrl_vs_stim", WRAPPER, outcome="deliberate", transforms=["logit"])
    return cases


def fixture_cases(rng):
    """The small cases whose inputs and speckle outputs the tests compare with."""
    case = two_group_case(rng, 10, 4, (300, 1500), "F")
    two_group_calls(case)
    other = three_group_case(rng, 6, ["A", "B", "C"] * 3, (300, 1500), "G", age=True)
    covariate_calls(other, ["logit"])
    weights = other.design.contrasts["B_vs_A"].tolist()
    other.add("B_vs_A", "B_vs_A", TTEST | {"contrast": weights}, transforms=["asin"])
    return {"fixture two groups": case, "fixture covariate": other}


def run_speckle(cases):
    """Write every case for R, run speckle, and read its outputs back."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, case in cases.items():
            directory = root / names[name]
            directory.mkdir()
            columns = ["sample_id", *case.variables]
            values = [np.asarray(case.samples[c]).tolist() for c in columns]
            lines = ["\t".join(columns)] + [
                "\t".join(map(str, row)) for row in zip(*values, strict=True)
            ]
            texts = {
                "samples.tsv": "\n".join(lines),
                "groups.txt": "\n".join(case.groups),
                "variables.json": json.dumps(case.variables),
                "calls.json": json.dumps([call for *_, call, _ in case.analyses]),
            }
            for file, text in texts.items():
                (directory / file).write_text(text + "\n")
            # Column by column, as R fills a samples x groups matrix.
            n_cells = np.ascontiguousarray(case.cube.n_cells.T, dtype="<f8")
            (directory / "n_cells.bin").write_bytes(n_cells.tobytes())
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        outputs = {
            name: [
                read_output(root / names[name], k) for k in range(len(case.analyses))
            ]
            for name, case in cases.items()
        }
        return outputs, (root / "versions.txt").read_text().split("\n")[:3]


def read_output(directory, k):
    """speckle's table of one call, by column, or its error message."""
    error = directory / f"call{k}_error.txt"
    if error.exists():
        return error.read_text().strip()
    lines = (directory / f"call{k}.tsv").read_text().splitlines()
    table = {}
    for column, *values in zip(*(line.split("\t") for line in lines), strict=True):
        if column in ("row", "BaselineProp.clusters"):
            table[column] = list(values)
        else:
            table[column] = np.array([float("nan" if v == "NA" else v) for v in values])
    return table


def difference(ours, reference, kind):
    """Largest |ours - reference| / scale; equal values, NaN included, differ by 0."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if kind == "p":
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    elif kind == "statistic":
        scale = np.maximum(np.abs(reference), 1.0)
    else:
        scale = np.abs(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.abs(ours - reference) / scale
    scaled[(ours == reference) | (np.isnan(ours) & np.isnan(reference))] = 0.0
    scaled[~np.isfinite(scaled)] = np.inf
    return float(np.max(scaled, initial=0.0))


def compare_table(table, reference, factors, common=False, statistics=False):
    """The largest differences in proportions (None if none) and in statistics.

    Infinite when speckle failed, the columns differ, or the cell types differ.
    With ``common``, the cell types of both are compared; with ``statistics``, only
    the statistics, p-values and FDRs.
    """
    if isinstance(reference, str) or isinstance(table, str):
        return np.inf, np.inf
    ours = table["group_id"].tolist()
    rows = [g for g in reference["row"] if g in ours] if common else reference["row"]
    if not common and sorted(ours) != sorted(rows):
        return np.inf, np.inf
    position = [ours.index(g) for g in rows]
    keep = [reference["row"].index(g) for g in rows]
    shown = {c for c in table.columns if c.startswith("mean_proportion_")}
    worst = {"": None, "p": 0.0, "statistic": 0.0}
    for column, values in reference.items():
        kind = KINDS.get(column, "")
        if column in ("row", "BaselineProp.clusters") or (statistics and not kind):
            continue
        if column.startswith("PropMean."):
            name = mean_column(column.removeprefix("PropMean."), table, factors)
            shown.discard(name)
        else:
            name = COLUMNS[column]
        if name not in table:
            return np.inf, np.inf
        value = difference(table[name][position], values[keep], kind)
        worst[kind] = max(worst[kind] or 0.0, value)
    if shown and not statistics:
        return np.inf, np.inf
    return worst[""], max(worst["p"], worst["statistic"])


def mean_column(label, table, factors):
    """CytePool's column for speckle's PropMean.<level> or PropMean.<factor><level>."""
    for factor in ["", *factors]:
        name = f"mean_proportion_{label.removeprefix(factor)}"
        if label.startswith(factor) and name in table:
            return name
    return f"mean_proportion_{label}"


FUNCTIONS = {
    "propeller": "propeller",
    "ttest": "propeller.ttest",
    "anova": "propeller.anova",
    "limma": "limma eBayes",
}


def compare(name, case, outputs):
    """Rows of (check, reference, metric, value, tolerance, result) for one case."""
    rows = []
    for (label, test, design, call, outcome), reference in zip(
        case.analyses, outputs, strict=True
    ):
        try:
            table = case.run(test, design, call)
        except ValueError as error:  # a refusal, which only a deliberate row expects
            table = str(error)
        robust = call["robust"] and not isinstance(table, str) and len(table) >= 10
        tolerance = CLOSE if robust else EXACT
        check = (f"{name}, {label}", FUNCTIONS[call["kind"]])
        if outcome == "pass":
            proportions, statistics = compare_table(table, reference, case.variables)
            passed = (proportions or 0.0) <= EXACT and statistics <= tolerance
            shown = "n/a" if proportions is None else f"{proportions:.1e}"
            value, bounds = (
                f"{shown}, {statistics:.1e}",
                f"{EXACT:.0e}, {tolerance:.0e}",
            )
            metric = "max scaled difference: proportions, statistics"
            rows.append(check + (metric, value, bounds, "pass" if passed else "FAIL"))
            continue
        expected, value = deliberate(table, reference, case.variables, tolerance)
        outcome = "deliberate difference" if expected else "FAIL"
        rows.append(check + ("behaviour", value, "n/a", outcome))
    return rows


def deliberate(table, reference, factors, tolerance):
    """Whether a deliberate difference behaves as documented, and how."""
    if isinstance(table, str):  # CytePool refuses data that speckle tests
        expected = "exactly" in table and not isinstance(reference, str)
        largest = (
            np.inf if isinstance(reference, str) else np.max(reference["Tstatistic"])
        )
        return expected, f"speckle: largest t {largest:.1e}; CytePool: refuses"
    if isinstance(reference, str):  # speckle stops; CytePool tests the contrast
        expected = "t" in table and "proportion_ratio" not in table
        return expected, f"speckle: {reference}; CytePool: t-test without ratio"
    if "proportion_ratio" not in table:  # speckle reports a ratio of nothing
        statistics = compare_table(table, reference, factors, statistics=True)[1]
        means = [v for column, v in reference.items() if column.startswith("PropMean")]
        product = np.prod(means, axis=0)  # its ratio, with the weights 1 and 1
        expected = statistics <= tolerance and np.allclose(
            reference["PropRatio"], product, rtol=1e-12
        )
        value = "speckle: PropRatio, the product of its PropMeans; CytePool: no ratio"
        return expected, value + f"; statistics agree to {statistics:.1e}"
    # speckle keeps an empty level, which changes its other values.
    statistics = compare_table(table, reference, factors, common=True)[1]
    ours = set(table["group_id"].tolist())
    extra = [group for group in reference["row"] if group not in ours]
    expected = statistics > tolerance and ours <= set(reference["row"])
    value = f"{statistics:.1e} on the cell types of both" + (
        f"; speckle also tests {', '.join(extra)}" if extra else ""
    )
    return expected, value


def write_fixtures(cases, outputs):
    """The inputs and speckle's tables of the fixture cases, in one file."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, case in cases.items():
        key = name.removeprefix("fixture ").replace(" ", "_")
        arrays[f"{key} n_cells"] = case.cube.n_cells
        arrays[f"{key} groups"] = np.array(case.groups)
        for column, values in case.samples.items():
            arrays[f"{key} {column}"] = np.asarray(values)
        for (label, *_), output in zip(case.analyses, outputs[name], strict=True):
            for column, values in output.items():
                arrays[f"{key} | {label} | {column}"] = np.asarray(values)
    np.savez_compressed(FIXTURE_DIRECTORY / "speckle.npz", **arrays)
    print(f"wrote speckle.npz to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(20261025))
    fixtures = fixture_cases(np.random.default_rng(20261026))
    if arguments.write_fixtures:
        cases |= fixtures
    outputs, versions = run_speckle(cases)
    print("Reference: " + ", ".join(versions))
    table = []
    for name, case in cases.items():
        table += compare(name, case, outputs[name])
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        write_fixtures(fixtures, {name: outputs[name] for name in fixtures})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
