"""Check the activities of regulators and pathways (ULM, MLM and the weighted
aggregate), their bar plot, and the networks CollecTRI and PROGENy against decoupler
and progeny.

Networks: CytePool downloads CollecTRI's regulons (Zenodo record 8192729) and the R
package progeny 1.28.0 (Bioconductor 3.20) into its cache. The files are compared with
the MD5 digests and sizes their sources publish (Zenodo's record, Bioconductor's
PACKAGES and the server's Content-Length); ``collectri(licence="academic")`` with
decoupler's ``op.collectri``; the commercial level with the licence purposes that
OmniPath's resources service lists, applied to the file by this script; ``progeny``'s
weights and p-values with progeny's ``model_human_full`` and ``model_mouse_full``, and
its ``top`` with progeny's ``getModel`` (100 human genes; 500 mouse genes, with ties);
decoupler's ``op.progeny``, which reads PROGENy through OmniPath with current gene
symbols, is reported for information.

Activities, on inputs simulated from fixed seeds: log-expression values of 6 samples
over CollecTRI's 6,693 targets and 2,000 other genes, the targets of 6 regulators
shifted in the last 3 samples, with 3 genes zero everywhere (left out by both); a
profile of t-statistics; the values with 3% missing in two samples (decoupler, which
refuses missing values, scores each profile over its observed genes); 6 samples over
PROGENy's targets and 1,000 other genes; a cube of one cell type (its log-CPM over the
genes that filterByExpr keeps); and limma_voom's moderated t-statistics in two cell
types that test different genes. ``ulm``, ``mlm`` and ``weighted_aggregate`` are
compared with decoupler 2.2.0's ``mt.ulm``, ``mt.mlm`` and ``mt.waggr``: scores,
estimates (``tval=False``), p-values before adjustment, and adjusted p-values (scipy's
Benjamini-Hochberg of decoupler's p-values; decoupler's own, in single precision);
MLM's p-values with scipy's ``t.sf`` of decoupler's t-statistics; the normalized
scores given decoupler's permutations (the genes in decoupler's order, and its seed);
the weighted sum of PROGENy's top 100 genes with progeny's ``progeny(scale =
FALSE)``; and ``plot_top_activities`` with decoupler's ``pl.barplot``. Deliberate
differences: the weighted aggregate's p-values; a sample whose values are all zero,
which decoupler leaves out; two sources with the same targets, for which decoupler's
MLM gives NaN t-statistics; and CollecTRI's interactions without a reference, which
decoupler's loader drops.

Tiers: exact, 1e-10, relative to the largest absolute value of a case for scores and
estimates and to each value for p-values; networks and figures identical; decoupler's
adjusted p-values within its single precision (1.2e-7). With ``--write-fixtures``,
smaller inputs (300 genes, 5 samples and a simulated network of 20 regulators), rows
in the format of CollecTRI's file and a package in the layout of progeny's, their data
simulated, are written to tests/data/activity for the tests.

Requirements:
    Python 3.13 with numpy, numba, scipy and matplotlib, run from the repository root
    so that ``cytepool`` imports from the working tree; access to the internet.
    decoupler 2.2 with pandas (``DECOUPLER_PYTHON``, by default this interpreter).
    R 4.4 or newer with progeny 1.28.0 and jsonlite (``RSCRIPT``, by default
    ``Rscript``).

Usage:
    PYTHONPATH=. python validation/activity.py
    PYTHONPATH=. python validation/activity.py --write-fixtures
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.figure
import numpy as np

from cytepool import (
    Cube,
    Design,
    Network,
    Table,
    collectri,
    limma_voom,
    mlm,
    progeny,
    ulm,
    weighted_aggregate,
)
from cytepool.normalization import filter_by_expression, log2_cpm, tmm_factors
from cytepool.plotting import plot_top_activities

FIXTURES = Path(__file__).resolve().parents[1] / "tests/data/activity"
EXACT = 1e-10
SINGLE = 1.2e-7  # 2^-23, the precision of decoupler's adjusted p-values
FUNCTIONS = {"ulm": ulm, "mlm": mlm, "waggr": weighted_aggregate}
ZENODO = "https://zenodo.org/api/records/8192729"
BIOCONDUCTOR = "https://bioconductor.org/packages/3.20/bioc/src/contrib/"
OMNIPATH = "https://omnipathdb.org/resources"
COLLECTRI_HEADER = ("source", "target", "weight", "resources", "references",
                    "sign_decision")  # fmt: skip
HEADERS = {
    "User-Agent": "CytePool validation (+https://github.com/NygenAnalytics/CytePool)"
}

DECOUPLER = r"""
import io, json, sys
import numpy as np, pandas as pd, scipy.stats
import decoupler as dc
import decoupler.op._collectri as loader

root = sys.argv[1]
spec = json.load(open(f"{root}/input.json"))
arrays = dict(np.load(f"{root}/input.npz"))
out, text = {}, {"version": dc.__version__, "cases": {}}
for case in spec["cases"]:
    data = spec["data"][case["data"]]
    df = pd.DataFrame(arrays[case["data"]], index=data["profiles"], columns=data["genes"])
    net = pd.DataFrame(spec["networks"][case["network"]])
    method, name, options = getattr(dc.mt, case["method"]), case["name"], case["options"]
    try:
        es, pv = method(df, net, tmin=case["tmin"], **options)
    except Exception as error:
        text["cases"][name] = {"error": f"{type(error).__name__}: {error}"}
        continue
    mat, obs, var = dc.pp.extract(df)
    sources, _, adj = dc.pp.adjmat(var, dc.pp.prune(var, net, tmin=case["tmin"]))
    raw_es, raw_p = method.func(mat, adj, **options)
    found = {"profiles": list(es.index), "sources": list(es.columns),
             "genes": list(map(str, var))}
    out[f"{name}/scores"], out[f"{name}/padj"] = es.to_numpy(), pv.to_numpy()
    out[f"{name}/raw_scores"], out[f"{name}/raw_p"] = raw_es, raw_p
    if case["method"] in ("ulm", "mlm"):
        out[f"{name}/estimates"] = method.func(mat, adj, tval=False)[0]
        df_resid = mat.shape[1] - (2 if case["method"] == "ulm" else adj.shape[1] + 1)
        out[f"{name}/t_sf"] = 2 * scipy.stats.t.sf(np.abs(raw_es), df_resid)
        if np.isfinite(raw_p).all():
            out[f"{name}/bh"] = scipy.stats.false_discovery_control(raw_p, axis=1)
    if case["method"] == "waggr" and options.get("times", 0) > 1:
        ridx = dc.mt._gsea._ridx(options["times"], mat.shape[1], options["seed"])
        out[f"{name}/permutations"] = ridx.astype(np.int16)
    if "bar" in case:
        fig = dc.pl.barplot(es, name=case["bar"], top=15, return_fig=True)
        ax = fig.axes[0]
        found["bar_names"] = [t.get_text() for t in ax.get_yticklabels()]
        out[f"{name}/bar_widths"] = np.array([p.get_width() for p in ax.containers[0]])
    text["cases"][name] = found
def rows(frame, columns):
    return [[str(v) if c != "weight" and c != "padj" else float(v) for v, c in
             zip(row, columns)] for row in frame[columns].itertuples(index=False)]
columns = ["source", "target", "weight", "resources", "references", "sign_decision"]
if spec["subset"]:
    data = open(spec["subset"], "rb").read()
    loader._download = lambda url, verbose=False: io.BytesIO(data)
    text["collectri"] = rows(dc.op.collectri(), columns)
else:
    academic = dc.op.collectri()
    text["collectri"] = rows(academic, columns)
    text["same_levels"] = bool(dc.op.collectri(license="commercial").equals(academic))
    text["progeny"] = rows(dc.op.progeny(), ["source", "target", "weight", "padj"])
json.dump(text, open(f"{root}/decoupler.json", "w"))
np.savez(f"{root}/decoupler.npz", **out)
"""

R_SCRIPT = r"""
suppressPackageStartupMessages({library(progeny); library(jsonlite)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
input <- fromJSON(file.path(root, "progeny_input.json"))
out <- list(version = paste("progeny", packageVersion("progeny")))
for (organism in c("human", "mouse")) {
  e <- new.env()
  data(list = paste0("model_", organism, "_full"), package = "progeny", envir = e)
  m <- e[[paste0("model_", organism, "_full")]]
  writeBin(as.double(m$weight), file.path(root, paste0(organism, "_weight.bin")))
  writeBin(as.double(m$p.value), file.path(root, paste0(organism, "_p.bin")))
  out[[organism]] <- list(gene = as.character(m$gene), pathway = as.character(m$pathway))
}
for (run in list(list("Human", 100), list("Mouse", 500))) {
  g <- getModel(run[[1]], top = run[[2]])
  long <- which(as.matrix(g) != 0, arr.ind = TRUE)
  out[[paste0("top_", run[[1]])]] <- list(gene = rownames(g)[long[, 1]],
    pathway = colnames(g)[long[, 2]], weight = as.matrix(g)[long])
}
x <- matrix(input$values, nrow = length(input$genes), dimnames = list(input$genes, NULL))
scores <- progeny(x, scale = FALSE, organism = "Human", top = 100)
out$scores <- as.vector(t(scores))
out$score_pathways <- colnames(scores)
write_json(out, file.path(root, "progeny.json"), digits = I(17), auto_unbox = TRUE)
"""


# Inputs.


def network_table(network):
    """A network's source, target and weight columns, as lists for JSON."""
    edges = network.edges
    return {name: edges[name].tolist() for name in ("source", "target", "weight")}


def expression(rng, genes, n_samples, network, regulators, shift=1.0):
    """Log-expression values, the targets of ``regulators`` shifted by their weights
    in the second half of the samples."""
    y = rng.normal(5.0, 1.5, len(genes)) + rng.normal(0.0, 1.0, (n_samples, len(genes)))
    position = {gene: j for j, gene in enumerate(genes)}
    for source in regulators:
        for target, weight in network[source].items():
            y[n_samples // 2 :, position[target]] += shift * weight
    return y


def inputs(small, networks):
    """The data (name to values, profiles, genes), the networks, the cases and the
    CytePool runs that are not plain arrays."""
    rng = np.random.default_rng(20261026)
    regulons = networks["collectri"]
    if small:
        # A network of CollecTRI's kind, simulated: 20 regulators of 6 to 30 of 240
        # genes, a fifth of the edges repressing.
        pool = [f"TARGET{i}" for i in range(240)]
        regulons = Network({
            f"TF{k}": {gene: rng.choice([1.0, 1.0, 1.0, 1.0, -1.0])
                       for gene in rng.choice(pool, rng.integers(6, 31), False)}
            for k in range(20)
        })  # fmt: skip
    targets = sorted(set(regulons.edges["target"].tolist()))
    genes = targets + [f"OTHER{i}" for i in range(60 if small else 2000)]
    genes = [genes[i] for i in rng.permutation(len(genes))]
    regulons = Network({s: {t: w for t, w in regulons[s].items() if t in genes}
                        for s in regulons})  # fmt: skip
    n = 5 if small else 6
    active = [s for s in regulons if len(regulons[s]) >= 20][:6]
    y = expression(rng, genes, n, regulons, active)
    y[:, :3] = 0.0  # genes zero in every sample, which both leave out
    statistic = (y[n // 2 :].mean(axis=0) - y[: n // 2].mean(axis=0)) / 0.4
    missing = y.copy()
    missing[:2][rng.random((2, len(genes))) < 0.03] = np.nan
    data = {
        "samples": (y, [f"s{i}" for i in range(n)], genes),
        "statistics": (statistic[np.newaxis], ["t"], genes),
    }
    nets = {"collectri": network_table(regulons)}
    flat = Network({s: dict.fromkeys(list(regulons[s]), 1.0) for s in regulons})
    nets["one_sign"] = network_table(flat)
    cases = [
        ("ulm", "samples", "collectri", "ulm", 5, {}),
        ("ulm, tmin 3, weights of one sign", "samples", "one_sign", "ulm", 3, {}),
        ("ulm, t-statistics", "statistics", "collectri", "ulm", 5, {}),
        ("mlm", "samples", "collectri", "mlm", 5, {}),
        ("weighted mean", "samples", "collectri", "waggr", 5, {"times": 0}),
        ("weighted sum", "samples", "collectri", "waggr", 5,
         {"times": 0, "fun": "wsum"}),
        ("weighted mean, 100 permutations", "samples", "collectri", "waggr", 5,
         {"times": 100, "seed": 42}),
    ]  # fmt: skip
    for row in range(2):
        observed = ~np.isnan(missing[row])
        name = f"missing_{row}"
        data[name] = (missing[row : row + 1, observed], [f"s{row}"],
                      [g for g, k in zip(genes, observed, strict=True) if k])  # fmt: skip
        cases += [(f"ulm, missing values, s{row}", name, "collectri", "ulm", 5, {}),
                  (f"mlm, missing values, s{row}", name, "collectri", "mlm", 5, {})]  # fmt: skip
    special = {"missing": missing}
    if not small:
        pathways = networks["progeny"]
        p_genes = sorted(set(pathways.edges["target"].tolist()))
        p_genes += [f"OTHER{i}" for i in range(1000)]
        z = expression(rng, p_genes, 6, pathways, ["EGFR", "TNFa"], 0.5)
        data["pathways"] = (z, [f"p{i}" for i in range(6)], p_genes)
        nets["progeny"] = network_table(pathways)
        zero = y[:3].copy()
        zero[1] = 0.0
        data["zero_sample"] = (zero, ["z0", "z1", "z2"], genes)
        twins = dict(regulons.items())
        twins["TWIN"] = dict(regulons[active[0]])
        nets["twins"] = network_table(Network(twins))
        cases += [
            ("mlm, PROGENy", "pathways", "progeny", "mlm", 5, {}),
            ("ulm, PROGENy", "pathways", "progeny", "ulm", 5, {}),
            ("weighted mean, PROGENy, 1,000 permutations", "pathways", "progeny",
             "waggr", 5, {"times": 1000, "seed": 42}),
            ("ulm, a sample of zeros", "zero_sample", "collectri", "ulm", 5, {}),
            ("mlm, two sources with the same targets", "samples", "twins", "mlm", 5,
             {}),
        ]  # fmt: skip
        special |= cube_and_result(rng, genes, data, cases)
    cases[0] = (*cases[0], "s4")  # its scores in a bar plot
    keys = ("name", "data", "network", "method", "tmin", "options", "bar")
    return data, nets, [dict(zip(keys, case)) for case in cases], special


def cube_and_result(rng, genes, data, cases):
    """A cube of two cell types and limma_voom's t-statistics in each, which test
    different genes; decoupler scores the log-CPM of one and each t-statistic."""
    means = np.exp(rng.uniform(1.0, 7.0, len(genes)))
    counts = rng.negative_binomial(10, 10 / (10 + means), (2, 8, len(genes)))
    counts[1, :, : len(genes) // 3] = 0  # genes not expressed in the T cells
    cube = Cube(
        {"counts": counts.astype(np.uint32)},
        samples={"sample_id": [f"S{i}" for i in range(8)],
                 "condition": ["ctrl", "stim"] * 4},
        groups={"group_id": ["B cells", "T cells"]},
        features={"feature_id": genes},
        n_cells=np.full((8, 2), 40),
    )  # fmt: skip
    kept = counts[0]
    keep = filter_by_expression(kept, sample_groups=np.zeros(8))
    values = log2_cpm(kept[:, keep], norm_factors=tmm_factors(kept[:, keep]),
                      prior_count=2.0)  # fmt: skip
    data["cube"] = (values, [f"S{i}" for i in range(8)],
                    [g for g, k in zip(genes, keep, strict=True) if k])  # fmt: skip
    cases.append(("ulm, a cube's cell type", "cube", "collectri", "ulm", 5, {}))
    result = limma_voom(cube, Design(cube.samples, "~ condition"),
                        contrasts={"stim": "conditionstim"})  # fmt: skip
    for group in result.groups:
        table = result.table(group, "stim")
        data[f"result_{group}"] = (table["t"][np.newaxis], [group],
                                   table["feature_id"].tolist())  # fmt: skip
        cases += [(f"ulm, limma_voom's t of the {group}", f"result_{group}",
                   "collectri", "ulm", 5, {}),
                  (f"mlm, limma_voom's t of the {group}", f"result_{group}",
                   "collectri", "mlm", 5, {})]  # fmt: skip
    return {"cube": cube, "result": result}


# The references.


def run_decoupler(root, data, nets, cases, subset=None):
    arrays = {name: values for name, (values, _, _) in data.items()}
    np.savez(root / "input.npz", **arrays)
    described = {name: {"profiles": p, "genes": g} for name, (_, p, g) in data.items()}
    payload = {"data": described, "networks": nets, "cases": cases,
               "subset": str(subset) if subset else None}  # fmt: skip
    (root / "input.json").write_text(json.dumps(payload))
    (root / "decoupler_program.py").write_text(DECOUPLER)
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    python = os.environ.get("DECOUPLER_PYTHON", sys.executable)
    subprocess.run([python, "-I", str(root / "decoupler_program.py"), str(root)],
                   check=True, env=environment)  # fmt: skip
    found = json.loads((root / "decoupler.json").read_text())
    return found, dict(np.load(root / "decoupler.npz"))


def run_r(root, genes, values):
    # R fills its genes x samples matrix by column: the genes of each sample in turn.
    payload = {"genes": genes, "values": values.ravel().tolist()}
    (root / "progeny_input.json").write_text(json.dumps(payload))
    (root / "progeny.R").write_text(R_SCRIPT)
    rscript = os.environ.get("RSCRIPT", "Rscript")
    command = f"ulimit -v 8000000; {rscript} --vanilla {root / 'progeny.R'} {root}"
    subprocess.run(["bash", "-c", command], check=True)
    found = json.loads((root / "progeny.json").read_text())
    for organism in ("human", "mouse"):
        for column in ("weight", "p"):
            path = root / f"{organism}_{column}.bin"
            found[organism][column] = np.fromfile(path, "<f8")
    return found


# Comparisons.


def scaled_difference(ours, theirs):
    """The largest difference over the largest absolute reference value; inf when
    NaN stands in different places."""
    if ours.shape != theirs.shape or not np.array_equal(np.isnan(ours),
                                                        np.isnan(theirs)):  # fmt: skip
        return np.inf
    return float(np.nanmax(np.abs(ours - theirs)) / np.nanmax(np.abs(theirs)))


def relative_difference(ours, theirs):
    """The largest difference relative to each reference value (at least 1e-300)."""
    if ours.shape != theirs.shape or not np.array_equal(np.isnan(ours),
                                                        np.isnan(theirs)):  # fmt: skip
        return np.inf
    floor = np.maximum(np.abs(theirs), 1e-300)
    return float(np.nanmax(np.abs(ours - theirs) / floor))


def single_difference(ours, theirs):
    """The largest difference of ours in single precision, relative to each of
    theirs, or to the smallest normal number of single precision below it."""
    rounded = ours.astype(np.float32).astype(np.float64)
    floor = np.maximum(np.abs(theirs), np.finfo(np.float32).tiny)
    return float(np.max(np.abs(rounded - theirs) / floor))


def run_case(case, data, nets, special):
    """CytePool's result of a case, and the rows of it that decoupler scored."""
    values, profiles, genes = data[case["data"]]
    network = Network(Table(nets[case["network"]]))
    options = {"min_targets": case["tmin"]}
    function = FUNCTIONS[case["method"]]
    if case["method"] == "waggr":
        times = case["options"]["times"]
        options |= {"statistic": "sum" if case["options"].get("fun") == "wsum"
                    else "mean", "n_permutations": times, "rng": 42}  # fmt: skip
    if case["data"] == "cube":
        found = function(special["cube"], network, group="B cells", **options)
        return found, list(range(values.shape[0]))
    if case["data"].startswith("result_"):
        group = case["data"].removeprefix("result_")
        found = function(special["result"], network, group=group, **options)
        return found, [0]
    if case["data"].startswith("missing_"):
        everything = data["samples"]
        found = function(special["missing"], network, feature_ids=everything[2],
                         sample_ids=everything[1], **options)  # fmt: skip
        return found, [int(case["data"].removeprefix("missing_"))]
    found = function(values, network, feature_ids=genes, sample_ids=profiles, **options)
    return found, list(range(values.shape[0]))


def check_case(case, found, arrays, data, nets, special, table):
    name = case["name"]
    reference = found["cases"][name]
    method = case["method"]
    if name.startswith("mlm, two sources"):
        try:
            run_case(case, data, nets, special)
            ours = "runs"
        except ValueError as error:
            ours = "refuses: " + str(error).split(":")[0]
        missing = int(np.isnan(arrays[f"{name}/raw_scores"]).sum())
        table.append((f"{name}: decoupler gives NaN t-statistics; CytePool {ours}",
                      "decoupler `mt.mlm`", "NaN t-statistics of decoupler",
                      float(missing), "deliberate"))  # fmt: skip
        return
    if method == "waggr" and case["options"]["times"]:
        # decoupler's permutations: the genes in its order and its seed.
        values, profiles, genes = data[case["data"]]
        position = {gene: j for j, gene in enumerate(genes)}
        order = [position[gene] for gene in reference["genes"]]
        result = weighted_aggregate(
            values[:, order], Network(Table(nets[case["network"]])),
            feature_ids=reference["genes"], sample_ids=profiles,
            n_permutations=case["options"]["times"], rng=case["options"]["seed"],
        )  # fmt: skip
        rows = list(range(values.shape[0]))
    else:
        result, rows = run_case(case, data, nets, special)
    if name == "ulm, a sample of zeros":
        ids = result.profiles["profile_id"].tolist()
        rows = [ids.index(p) for p in reference["profiles"]]
        table.append((f"{name}: decoupler leaves out the sample z1 of zeros; CytePool "
                      "gives it NaN scores", "decoupler", "NaN scores of z1",
                      float(np.isnan(result.scores[ids.index("z1")]).mean()),
                      "deliberate"))  # fmt: skip
    columns = [list(result.sources).index(s) for s in reference["sources"]
               if s in result.sources]  # fmt: skip
    scored = ~np.all(np.isnan(result.scores[rows]), axis=0)
    ours = sorted(s for s, kept in zip(result.sources, scored, strict=True) if kept)
    if ours != sorted(reference["sources"]):
        table.append((f"{name}: the sources scored", "decoupler", "the same", np.inf,
                      0.0))  # fmt: skip
        return
    pick = np.ix_(rows, columns)
    shape = f"{len(rows)} x {len(columns)}"
    if method == "waggr":
        label = "normalized scores, decoupler's permutations" if case["options"][
            "times"] else "scores"  # fmt: skip
        theirs = arrays[f"{name}/raw_scores"]
        table.append((f"{name}: {label} ({shape})", "decoupler `mt.waggr`",
                      "largest difference / largest value",
                      scaled_difference(result.scores[pick], theirs), EXACT))  # fmt: skip
        if case["options"]["times"]:
            raw = arrays[f"{name}/raw_p"]
            table.append((f"{name}: p-values", "decoupler `mt.waggr`",
                          "largest difference",
                          float(np.max(np.abs(result.p_value[pick] - raw))),
                          "deliberate"))  # fmt: skip
        return
    family = f"decoupler `mt.{method}`"
    table.append((f"{name}: t-statistics ({shape})", family, "largest difference / largest value",
                  scaled_difference(result.scores[pick],
                                    arrays[f"{name}/raw_scores"]), EXACT))  # fmt: skip
    table.append((f"{name}: estimates", family, "largest difference / largest value",
                  scaled_difference(result.estimates[pick],
                                    arrays[f"{name}/estimates"]), EXACT))  # fmt: skip
    p_reference = arrays[f"{name}/raw_p"] if method == "ulm" else arrays[f"{name}/t_sf"]
    table.append((f"{name}: p-values", "decoupler" if method == "ulm" else
                  "scipy `t.sf` of decoupler's t", "max relative difference",
                  relative_difference(result.p_value[pick], p_reference), EXACT))  # fmt: skip
    bh = arrays.get(f"{name}/bh") if method == "ulm" else None
    if bh is not None and bh.shape[1] > 1:
        table.append((f"{name}: adjusted p-values", "scipy's Benjamini-Hochberg",
                      "max relative difference",
                      relative_difference(result.adjusted_p_value[pick], bh),
                      EXACT))  # fmt: skip
        table.append((f"{name}: adjusted p-values", "decoupler (single precision)",
                      "max relative difference",
                      single_difference(result.adjusted_p_value[pick],
                                        arrays[f"{name}/padj"]), SINGLE))  # fmt: skip
    if "bar_names" in reference:
        ax = plot_top_activities(result, case["bar"], top=15,
                                 ax=matplotlib.figure.Figure().subplots())  # fmt: skip
        names = [label.get_text() for label in ax.get_yticklabels()]
        widths = np.array([bar.get_width() for bar in ax.patches])
        same = names == reference["bar_names"]
        table.append((f"plot_top_activities: the 15 bars of {case['bar']}",
                      "decoupler `pl.barplot`", "largest difference / largest bar",
                      scaled_difference(widths, arrays[f"{name}/bar_widths"])
                      if same else np.inf, EXACT))  # fmt: skip


def check_networks(found, r_found, table, networks):
    """The parsed networks against decoupler, OmniPath's licences and progeny."""
    ours = networks["academic"].edges
    columns = ("source", "target", "weight", "resources", "references",
               "sign_decision")  # fmt: skip
    mine = {
        tuple(row) for row in zip(*(ours[c].tolist() for c in columns), strict=True)
    }
    theirs = {tuple(row) for row in found["collectri"]}
    extra = mine - theirs
    table.append(("CollecTRI, academic level: the interactions decoupler reads, all "
                  "columns", "decoupler `op.collectri`", "rows that differ",
                  float(len(theirs - mine)), 0.0))  # fmt: skip
    table.append((f"CollecTRI: the {len(extra)} interactions without a reference, "
                  "which decoupler drops (pandas' dropna)", "decoupler `op.collectri`",
                  "of them with an empty reference",
                  float(sum(1 for row in extra if row[4] == "")), "deliberate"))  # fmt: skip
    table.append(("decoupler's op.collectri gives the same table at its levels "
                  "commercial and academic", "decoupler", "same",
                  float(found["same_levels"]), "information"))  # fmt: skip
    resources = json.loads(fetch(OMNIPATH))
    academic = {name.removesuffix("_CollecTRI").replace("_", "")
                for name, entry in resources.items()
                if name.endswith("_CollecTRI")
                and entry["license"]["purpose"] != "commercial"}  # fmt: skip
    expected = {row for row in mine if set(row[3].split(";")) - academic}
    commercial = networks["commercial"].edges
    kept = {tuple(row) for row in zip(*(commercial[c].tolist() for c in columns),
                                      strict=True)}  # fmt: skip
    table.append((f"CollecTRI, commercial level: {len(kept):,} interactions, without "
                  f"those of {sorted(academic)} alone", "OmniPath's resource licences",
                  "rows that differ", float(len(kept ^ expected)), 0.0))  # fmt: skip
    for organism in ("human", "mouse"):
        full = networks[f"progeny_{organism}_full"].edges
        pairs = {(s, t): (w, p) for s, t, w, p in zip(
            full["source"].tolist(), full["target"].tolist(), full["weight"].tolist(),
            full["p_value"].tolist(), strict=True)}  # fmt: skip
        model = r_found[organism]
        reference = {(p, g): (w, q) for g, p, w, q in zip(
            model["gene"], model["pathway"], model["weight"].tolist(),
            model["p"].tolist(), strict=True)}  # fmt: skip
        table.append((f"PROGENy, {organism}: {len(reference):,} genes and pathways, "
                      "their weights and p-values", f"progeny `model_{organism}_full`",
                      "entries that differ",
                      float(len(set(pairs.items()) ^ set(reference.items()))), 0.0))  # fmt: skip
    for organism, top in (("Human", 100), ("Mouse", 500)):
        got = networks[f"progeny_{organism.lower()}_{top}"].edges
        mine_top = set(zip(got["source"].tolist(), got["target"].tolist(),
                           got["weight"].tolist(), strict=True))  # fmt: skip
        ref = r_found[f"top_{organism}"]
        theirs_top = set(zip(ref["pathway"], ref["gene"], ref["weight"], strict=True))
        table.append((f"PROGENy, {organism.lower()}, top {top} ({len(mine_top):,} "
                      "edges, ties kept)", "progeny `getModel`", "edges that differ",
                      float(len(mine_top ^ theirs_top)), 0.0))  # fmt: skip
    served = {(s, t): (w, p) for s, t, w, p in found["progeny"]}
    ours_p = networks["progeny_human"].edges
    pairs = {(s, t): (w, p) for s, t, w, p in zip(
        ours_p["source"].tolist(), ours_p["target"].tolist(),
        ours_p["weight"].tolist(), ours_p["p_value"].tolist(), strict=True)}  # fmt: skip
    shared = set(pairs) & set(served)
    alike = sum(1 for key in shared if pairs[key] == served[key])
    table.append((f"PROGENy through OmniPath (current gene symbols): {len(served):,} "
                  f"edges, {len(pairs):,} here, {len(shared):,} named alike, "
                  f"{alike:,} of them with the same weight and p-value",
                  "decoupler `op.progeny`", "share named alike",
                  len(shared) / len(pairs), "information"))  # fmt: skip


def fetch(url):
    """The bytes at ``url``."""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as answer:
        return answer.read()


def check_files(table):
    """The cached files against the digests and sizes their sources publish."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = cache / "cytepool" / "networks"
    record = json.loads(fetch(ZENODO))
    entry = next(f for f in record["files"] if f["key"] == "CollecTRI_regulons.csv")
    packages = fetch(BIOCONDUCTOR + "PACKAGES")
    block = packages.decode().split("Package: progeny\n")[1].split("\n\n")[0]
    md5 = next(line.split(": ")[1] for line in block.splitlines()
               if line.startswith("MD5sum"))  # fmt: skip
    head = urllib.request.Request(
        BIOCONDUCTOR + "progeny_1.28.0.tar.gz", method="HEAD", headers=HEADERS
    )
    size = int(urllib.request.urlopen(head, timeout=120).headers["Content-Length"])
    for file, digest, published in (
        ("CollecTRI_regulons.csv", entry["checksum"].removeprefix("md5:"),
         entry["size"]),
        ("progeny_1.28.0.tar.gz", md5, size),
    ):  # fmt: skip
        data = (directory / file).read_bytes()
        same = hashlib.md5(data).hexdigest() == digest and len(data) == published
        table.append((f"{file}: {len(data):,} bytes, MD5 {digest}", "the published "
                      "record", "differs", float(not same), 0.0))  # fmt: skip


def check_progeny_scores(r_found, values, genes, network, table):
    result = weighted_aggregate(values, network, feature_ids=genes, statistic="sum",
                                n_permutations=0, min_targets=1)  # fmt: skip
    theirs = np.array(r_found["scores"]).reshape(values.shape[0], -1)
    columns = [list(result.sources).index(s) for s in r_found["score_pathways"]]
    table.append(("weighted sum of PROGENy's top 100 genes (6 samples x 14 pathways)",
                  "progeny `progeny(scale = FALSE)`", "largest difference / largest value",
                  scaled_difference(result.scores[:, columns], theirs), EXACT))  # fmt: skip


# Fixtures.


def write_fixtures(data, nets, cases, found, arrays, subset_rows, network_files):
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    saved = {name: values for name, (values, _, _) in data.items()}
    saved |= {key: value for key, value in arrays.items()}
    np.savez_compressed(FIXTURES / "decoupler.npz", **saved)
    described = {name: {"profiles": p, "genes": g} for name, (_, p, g) in data.items()}
    fixture = {"data": described, "networks": nets, "cases": cases,
               "found": found["cases"], "collectri": found["collectri"]}  # fmt: skip
    (FIXTURES / "decoupler.json").write_text(json.dumps(fixture, separators=(",", ":")))
    (FIXTURES / "collectri_rows.csv").write_bytes(subset_rows)
    for name, payload in network_files.items():
        (FIXTURES / name).write_bytes(payload)
    print(f"wrote the fixtures to {FIXTURES}")


def collectri_rows():
    """Rows in the format of CollecTRI's file, simulated: regulators AP1, NFKB and 8
    others; each row with one to three resources, one or two references, or none; and
    rows that GEREDB alone lists."""
    rng = np.random.default_rng(3)
    resources = ["ExTRI_CollecTRI", "TRRUST_CollecTRI", "NTNU.Curated_CollecTRI",
                 "SIGNOR_CollecTRI", "DoRothEA-A_CollecTRI", "HTRI_CollecTRI"]  # fmt: skip
    signs = ["PMID", "default activation", "regulon"]
    rows = [list(COLLECTRI_HEADER)]
    for i in range(80):
        listed = ["CollecTRI", *rng.choice(resources, rng.integers(1, 4), False)]
        if i % 10 == 9:
            listed = ["CollecTRI", "GEREDB_CollecTRI"]
        references = ";".join(f"CollecTRI:{p}" for p in
                              rng.integers(1_000_000, 40_000_000, i % 3))  # fmt: skip
        source = ["AP1", "NFKB", *(f"TF{k}" for k in range(8))][i % 10]
        rows.append([source, f"GENE{i}", "-1.0" if i % 7 == 3 else "1.0",
                     ";".join(listed), references, signs[i % 3]])  # fmt: skip
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue().encode()


def progeny_fixture(root):
    """A small package in the layout of progeny's, its data simulated: models of 300
    genes and pathways saved by R with xz, the human genes a factor and the mouse genes
    text, the human model with a tie at the cutoff of its top 3; and dplyr's top 3."""
    script = r"""
    root <- commandArgs(trailingOnly = TRUE)[[1]]
    set.seed(1)
    pathways <- c("Androgen", "EGFR", "Estrogen", "Hypoxia", "JAK-STAT", "MAPK",
                  "NFkB", "p53", "PI3K", "TGFb", "TNFa", "Trail", "VEGF", "WNT")
    for (organism in c("human", "mouse")) {
      name <- paste0("model_", organism, "_full")
      genes <- sprintf(if (organism == "human") "GENE%04d" else "Gene%04d",
                       sample(5000, 300))
      m <- data.frame(gene = factor(genes),
                      pathway = factor(sample(pathways, 300, replace = TRUE)),
                      weight = rnorm(300, sd = 2), p.value = runif(300))
      if (organism == "mouse") m$gene <- as.character(m$gene)
      if (organism == "human") {
        # A tie at the cutoff of the top 3: the 3rd and 4th p-values of a pathway.
        first <- which(m$pathway == m$pathway[1])
        ranked <- first[order(m$p.value[first])]
        m$p.value[ranked[4]] <- m$p.value[ranked[3]]
        top <- as.data.frame(dplyr::top_n(dplyr::group_by(m, pathway), 3,
                                          wt = -p.value))
        write.csv(data.frame(gene = as.character(top$gene),
                             pathway = as.character(top$pathway)),
                  file.path(root, "top3.csv"), row.names = FALSE)
      }
      assign(name, m)
      save(list = name, file = file.path(root, paste0(name, ".rda")), compress = "xz")
      exact <- transform(m, weight = sprintf("%.17g", weight),
                         p.value = sprintf("%.17g", p.value))
      write.csv(exact, file.path(root, paste0(name, ".csv")), row.names = FALSE)
    }
    """
    (root / "fixture.R").write_text(script)
    rscript = os.environ.get("RSCRIPT", "Rscript")
    subprocess.run(["bash", "-c", f"ulimit -v 8000000; {rscript} --vanilla "
                    f"{root / 'fixture.R'} {root}"], check=True)  # fmt: skip
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for organism in ("human", "mouse"):
            path = root / f"model_{organism}_full.rda"
            info = tar.gettarinfo(path, f"progeny/data/{path.name}")
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with path.open("rb") as file:
                tar.addfile(info, file)
    files = {"progeny_package.tar.gz": archive.getvalue(),
             "progeny_top3.csv": (root / "top3.csv").read_bytes()}  # fmt: skip
    for organism in ("human", "mouse"):
        name = f"model_{organism}_full.csv"
        files[f"progeny_{name}"] = (root / name).read_bytes()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true")
    small = parser.parse_args().write_fixtures
    networks = {
        "academic": collectri(licence="academic"),
        "commercial": collectri(),
        "progeny_human": progeny(),
        "progeny_human_full": progeny(max_p_value=None),
        "progeny_mouse_full": progeny(organism="mouse", max_p_value=None),
        "progeny_human_100": progeny(top=100, max_p_value=None),
        "progeny_mouse_500": progeny(organism="mouse", top=500, max_p_value=None),
    }
    pathways = progeny(top=500)
    data, nets, cases, special = inputs(
        small, {"collectri": networks["academic"], "progeny": pathways}
    )
    table = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        subset = collectri_rows() if small else None
        if small:
            (root / "subset.csv").write_bytes(subset)
        found, arrays = run_decoupler(root, data, nets, cases,
                                      root / "subset.csv" if small else None)  # fmt: skip
        top100 = networks["progeny_human_100"]
        genes = sorted(set(top100.edges["target"].tolist())) + ["NOT_A_TARGET"]
        values = np.random.default_rng(5).normal(6.0, 2.0, (6, len(genes)))
        r_found = run_r(root, genes, values)
        files = progeny_fixture(root) if small else {}
    print(f"Reference: decoupler {found['version']}, {r_found['version']}")
    for case in cases:
        check_case(case, found, arrays, data, nets, special, table)
    if not small:
        check_networks(found, r_found, table, networks)
        check_files(table)
    check_progeny_scores(r_found, values, genes, networks["progeny_human_100"], table)
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
        print(f"| {check} | {source} | {metric} | {value:.3g} | {shown} | {result} |")
    print(f"{len(table)} checks, {failures} failed")
    if small:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(data, nets, cases, found, arrays, subset, files)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
