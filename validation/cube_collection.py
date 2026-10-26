"""Check CubeCollection against anndata's concat, pandas and limma run study by study.

The script simulates the cells of three studies of one stimulation (seed ``SEED``)
and pools each with ``from_cells``:

- ``lupus``: 8 samples (donors D1 to D4, each control and stimulated), 4 cell types
  with Cell Ontology ids, 700 genes;
- ``covid``: 10 samples (donors D1 to D5, paired: the donor and sample ids of
  ``lupus`` again), 5 cell types, two of which share a Cell Ontology id and are
  merged, 750 genes, of which 40 are filled with zeros and marked not measured;
- ``atlas``: 6 samples of 6 donors with integer ids, unpaired, 5 cell types without
  ontology ids, aligned by a mapping that leaves one out, 950 genes.

The collection aligns the groups by ``cell_ontology_id``, with a group mapping for
``atlas``, and writes donor ids ``study::donor`` (``within_study=["donor"]``). Some
sample and cell type entries have fewer than 10 cells, and one none.

References, run as separate programs on files in a temporary directory:

- anndata and pandas: each study's cube is exported with ``to_anndata``; the
  reference drops the genes that ``var["measured"]`` marks as not measured, gives
  each row its aligned group (the ontology id, or the mapping), sums the rows of each
  sample and aligned group, and joins the studies with ``anndata.concat``
  (``label="study"``). The outer join must give the collection's genes (as a set, and
  in the order in which the studies first list them, from ``pandas.unique``),
  ``measured`` (where its dense layers are not NaN) and the values of every aligned
  cube; the inner join, kept to the cell types of every study, must give ``stack()``:
  genes, groups, samples, every layer and the cell numbers. ``pandas.concat`` of the
  sample tables, with sample and donor ids written ``study::id``, must give
  ``samples``.
- limma 3.62.1 and edgeR 4.4.0 in R: limma-voom on each study's own counts, summing
  the groups that share an aligned id, and on the stacked counts of ``lupus`` and
  ``covid`` with ``~ donor + condition``, the steps of
  ``validation/differential_expression.py`` (``min_cells=10``, ``filterByExpr``,
  TMM, ``voom`` with ``adaptive.span = TRUE``, ``lmFit``, ``eBayes``, ``topTable``).
  ``limma_voom`` on every aligned cube, lined up by ``align_study_results``, must give
  limma's tables, ``"not_measured"`` for every gene the study did not measure and
  ``"not_tested"`` for the others limma left out; ``limma_voom`` on the stack must
  give the joint model. A control fits the joint model with the donor ids as the
  studies give them, so that donor D1 of both studies is taken for one person: its
  results must differ.

Tolerances: identical ids, orders, masks and integers; a relative 1e-10 for float
layers and for limma's statistics, which differences are scaled as in
``validation/differential_expression.py`` (by max(|reference|, 1), and p-values by
max(p, 1e-300) max(1, |log p|)).

Requirements:
    This script: numpy, pandas, anndata and h5py, with cytepool from the source tree.
    Reference program: anndata and pandas, in REFERENCE_PYTHON (default: this
    interpreter). R 4.4 with limma, edgeR and jsonlite: set RSCRIPT to its Rscript
    when it is not on PATH. R runs with its address space limited to 8 GB.

Usage, from the repository root:
    PYTHONPATH=. python validation/cube_collection.py
    PYTHONPATH=. python validation/cube_collection.py --write-fixtures
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from cytepool.cube import CubeCollection, from_cells
from cytepool.design import Design
from cytepool.differential_expression import limma_voom
from cytepool.io.anndata import to_anndata
from cytepool.meta_analysis import align_study_results

SEED = 20261026
EXACT = 1e-10
MEMORY = 8_000_000 * 1024  # bytes of address space for R
FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "cube_collection"
)
LAYERS = ("counts", "detected", "log_sum", "log_sum_sq", "linear_sum")
TABLE = ("logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B", "SE")
COLUMNS = (
    "log2_fold_change",
    "average_log2_cpm",
    "t",
    "p_value",
    "adjusted_p_value",
    "b_statistic",
    "standard_error",
)
ONTOLOGY = {"B": "CL:0000236", "CD4": "CL:0000624", "CD8": "CL:0000625"}
MONOCYTE, NK = "CL:0000860", "CL:0000623"
MAPPING = {
    "atlas": {
        "Bcells": ONTOLOGY["B"],
        "CD4": ONTOLOGY["CD4"],
        "CD8": ONTOLOGY["CD8"],
        "Mono": MONOCYTE,
        "Doublets": None,
    }
}
FORMULAS = {"lupus": "~ donor + condition", "covid": "~ donor + condition"}
FORMULAS["atlas"] = "~ condition"

REFERENCE_PROGRAM = textwrap.dedent(
    '''
    """anndata's concat of the aligned studies, and pandas' concat of their samples."""

    import json
    import sys
    import warnings
    from importlib.metadata import version

    import anndata
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    root = sys.argv[1]
    spec = json.load(open(f"{root}/spec.json"))
    sample_tables = json.load(open(f"{root}/samples.json"))  # NaN marks missing values
    layers = ("counts", "detected", "log_sum", "log_sum_sq", "linear_sum")
    aligned = {}
    for name in spec["studies"]:
        adata = anndata.read_h5ad(f"{root}/{name}.h5ad")
        if "measured" in adata.var:
            adata = adata[:, adata.var["measured"].to_numpy(bool)]
        mapping = spec["mapping"].get(name, {})
        groups = adata.obs["group_id"].astype(str).tolist()
        ontology = (
            adata.obs["cell_ontology_id"].astype(object).tolist()
            if "cell_ontology_id" in adata.obs
            else [None] * len(groups)
        )
        keys = [mapping[g] if g in mapping else o for g, o in zip(groups, ontology)]
        rows = np.array([key is not None for key in keys])
        pairs = pd.MultiIndex.from_arrays(
            [adata.obs["sample_id"].astype(str)[rows], np.array(keys, object)[rows]]
        )
        codes, unique = pd.factorize(pairs)
        summed = {}
        for layer in layers:
            values = np.asarray(adata.layers[layer], dtype=np.float64)[rows]
            summed[layer] = np.zeros((len(unique), adata.n_vars))
            np.add.at(summed[layer], codes, values)
        cells = np.bincount(codes, weights=adata.obs["n_cells"].to_numpy()[rows])
        obs = pd.DataFrame(
            {
                "sample_id": unique.get_level_values(0).astype(str),
                "group": unique.get_level_values(1).astype(str),
                "n_cells": cells,
            },
            index=[f"{s}|{g}" for s, g in unique],
        )
        aligned[name] = anndata.AnnData(
            obs=obs, var=pd.DataFrame(index=adata.var_names), layers=summed
        )
    out = {}
    for join in ("outer", "inner"):
        joined = anndata.concat(
            aligned, join=join, label="study", index_unique="::"
        )
        if join == "inner":
            every = set.intersection(*(set(a.obs["group"]) for a in aligned.values()))
            joined = joined[joined.obs["group"].isin(every).to_numpy()]
        out[f"{join}_genes"] = joined.var_names.to_numpy().astype(str)
        for key in ("study", "sample_id", "group"):
            out[f"{join}_{key}"] = np.array(joined.obs[key].astype(str).tolist())
        out[f"{join}_n_cells"] = joined.obs["n_cells"].to_numpy()
        for layer in layers:
            out[f"{join}_{layer}"] = np.asarray(joined.layers[layer], np.float64)
    out["first_listed"] = pd.unique(
        np.concatenate([a.var_names.to_numpy() for a in aligned.values()])
    ).astype(str)
    np.savez(f"{root}/reference.npz", **out)
    frames = []
    for name in spec["studies"]:
        columns = sample_tables[name]
        frame = pd.DataFrame(
            {c: np.array(v["values"], dtype=v["dtype"]) for c, v in columns.items()}
        )
        for column in ["sample_id", *spec["within_study"]]:
            if column in frame:
                frame[column] = name + "::" + frame[column].astype(str)
        frames.append(frame.assign(study=name))
    samples = pd.concat(frames, join="outer", ignore_index=True)
    table = {
        column: {
            "dtype": str(samples[column].dtype),
            "values": [
                None if pd.isna(value) else value
                for value in samples[column].astype(object).tolist()
            ],
        }
        for column in samples.columns
    }
    versions = {p: version(p) for p in ("anndata", "pandas", "numpy")}
    versions["python"] = sys.version.split()[0]
    json.dump(
        {"samples": table, "versions": versions},
        open(f"{root}/reference.json", "w"),
        default=float,
    )
    '''
)

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR); library(jsonlite)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
spec <- fromJSON(file.path(root, "spec.json"), simplifyVector = FALSE)
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, path) writeBin(as.double(x), path, size = 8)
study <- function(name) {
  dir <- file.path(root, "r", name)
  genes <- readLines(file.path(dir, "genes.txt"))
  groups <- readLines(file.path(dir, "groups.txt"))
  samples <- read.delim(file.path(dir, "samples.tsv"), colClasses = "character")
  # The aligned id of a group: the study's mapping, else its Cell Ontology id.
  ontology <- readLines(file.path(dir, "ontology.txt"))
  mapping <- spec$mapping[[name]]
  aligned <- vapply(seq_along(groups), function(k) {
    if (!is.null(mapping) && groups[k] %in% names(mapping)) {
      m <- mapping[[groups[k]]]; if (is.null(m)) NA_character_ else m
    } else ontology[k]
  }, "")
  list(name = name, genes = genes, samples = samples, aligned = aligned,
       measured = readLines(file.path(dir, "measured.txt")) == "TRUE",
       counts = array(rd(file.path(dir, "counts.bin")),
                      dim = c(length(genes), nrow(samples), length(groups))),
       cells = matrix(rd(file.path(dir, "n_cells.bin")), nrow = nrow(samples)))
}
for (a in spec$analyses) {
  data <- lapply(unlist(a$studies), study)
  # One study: all its genes and aligned groups; several: those every study has.
  genes <- if (length(data) == 1) data[[1]]$genes else
    Reduce(intersect, lapply(data, function(d) d$genes[d$measured]))
  groups <- Reduce(intersect, lapply(data, function(d) unique(na.omit(d$aligned))))
  for (g in groups) {
    y <- NULL; cells <- NULL; samples <- NULL
    for (d in data) {
      members <- which(d$aligned == g)
      part <- d$counts[match(genes, d$genes), , members, drop = FALSE]
      y <- cbind(y, apply(part, c(1, 2), sum))
      cells <- c(cells, rowSums(d$cells[, members, drop = FALSE]))
      s <- d$samples
      s$donor <- paste0(d$name, "::", s$donor)
      samples <- rbind(samples, s)
    }
    samples$donor <- factor(samples$donor, levels = sort(unique(samples$donor)))
    samples$condition <- factor(samples$condition, levels = c("ctrl", "stim"))
    used <- cells >= 10 & colSums(y) > 0
    X <- model.matrix(as.formula(a$formula), samples[used, , drop = FALSE])
    dge <- DGEList(y[, used, drop = FALSE])
    rownames(dge) <- genes
    dge <- dge[filterByExpr(dge, design = X), , keep.lib.sizes = FALSE]
    dge <- calcNormFactors(dge)
    fit <- eBayes(lmFit(voom(dge, X, adaptive.span = TRUE), X))
    tt <- topTable(fit, coef = "conditionstim", number = Inf, sort.by = "none")
    se <- sqrt(fit$s2.post) * fit$stdev.unscaled[, "conditionstim"]
    out <- file.path(root, "r", paste0(a$name, "__", gsub(":", "_", g)))
    dir.create(out)
    writeLines(rownames(tt), file.path(out, "genes.txt"))
    wr(cbind(as.matrix(tt[, c("logFC", "AveExpr", "t", "P.Value", "adj.P.Val",
                              "B")]), se), file.path(out, "table.bin"))
  }
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR"))), file.path(root, "versions.txt"))
"""


def simulate(rng, name, genes, groups, samples, cells, unmeasured=()):
    """Pool simulated cells of one study: a profile per cell type, a shift per
    donor, and a stimulation that changes a tenth of the genes."""
    n = len(genes)
    stim = np.where(rng.random(n) < 0.1, rng.normal(0.0, 1.2, n), 0.0)
    shift = {d: rng.normal(0.0, 0.3, n) for d in dict.fromkeys(samples["donor"])}
    base = rng.normal(0.0, 1.5, n)
    blocks, sample_labels, group_labels = [], [], []
    for g, group in enumerate(groups["group_id"]):
        profile = base + rng.normal(0.0, 0.7, n)
        for s, sample in enumerate(samples["sample_id"]):
            log_mean = profile + shift[samples["donor"][s]]
            log_mean = log_mean + stim * (samples["condition"][s] == "stim")
            share = np.exp(log_mean) / np.exp(log_mean).sum()
            depth = rng.uniform(1000.0, 4000.0, (cells[s, g], 1))
            counts = rng.poisson(rng.gamma(4.0, 0.25, (cells[s, g], n)) * depth * share)
            counts[:, list(unmeasured)] = 0
            empty = np.flatnonzero(counts.sum(axis=1) == 0)
            counts[empty, n - 1] = 1
            blocks.append(counts)
            sample_labels += [sample] * cells[s, g]
            group_labels += [group] * cells[s, g]
    measured = np.ones(n, dtype=bool)
    measured[list(unmeasured)] = False
    return from_cells(
        np.vstack(blocks),
        samples=sample_labels,
        groups=group_labels,
        sample_table=samples,
        group_table=groups,
        features={"feature_id": genes, "measured": measured},
        source={"study": name, "simulation": "validation/cube_collection.py"},
    )


def build(seed, sizes):
    """The three studies' cubes, from ``sizes`` = (genes, cells per entry)."""
    rng = np.random.default_rng(seed)
    n, per_entry = sizes
    universe = [f"ENSG{k:011d}" for k in range(1, n + 1)]
    tenth = n // 10
    paired = {
        "lupus": {"donors": 4, "genes": (0, 7 * tenth), "age": np.int64},
        "covid": {"donors": 5, "genes": (tenth, 17 * tenth // 2), "age": np.float64},
    }
    cubes = {}
    for name, spec in paired.items():
        donors = [f"D{k + 1}" for k in range(spec["donors"]) for _ in range(2)]
        samples = {
            "sample_id": [f"S{k + 1}" for k in range(len(donors))],
            "donor": donors,
            "condition": ["ctrl", "stim"] * spec["donors"],
            "age": np.array([30 + 3 * k for k in range(len(donors))], spec["age"]),
        }
        if name == "lupus":
            samples["sex"] = ["female", "female", "male", "male"] * 2
            groups = {
                "group_id": ["B cell", "CD4 T cell", "CD8 T cell", "monocyte"],
                "cell_ontology_id": [*ONTOLOGY.values(), MONOCYTE],
            }
        else:
            samples["age"][3] = np.nan
            samples["batch"] = ["b1"] * 4 + ["b2"] * 6
            # Two kinds of CD4 T cells, which share an ontology id and merge.
            groups = {
                "group_id": ["B", "T4 naive", "T4 memory", "T8", "NK"],
                "cell_ontology_id": [
                    ONTOLOGY["B"], ONTOLOGY["CD4"], ONTOLOGY["CD4"], ONTOLOGY["CD8"], NK
                ],
            }  # fmt: skip
        cells = rng.poisson(per_entry, (len(donors), len(groups["group_id"])))
        cells[1, 0] = 4  # too few cells for limma_voom's min_cells
        start, stop = spec["genes"]
        unmeasured = range(1, 6 * tenth - 1, 15) if name == "covid" else ()
        cubes[name] = simulate(
            rng, name, universe[start:stop], groups, samples, cells, unmeasured
        )
    samples = {
        "sample_id": [f"S{k + 1}" for k in range(6)],
        "donor": np.arange(101, 107),
        "condition": ["ctrl", "stim"] * 3,
        "age": np.arange(40, 46),
    }
    groups = {"group_id": list(MAPPING["atlas"])}
    cells = rng.poisson(per_entry, (6, 5))
    cells[2, 1] = 0  # an entry without cells, of a cell type every study has
    cubes["atlas"] = simulate(
        rng, "atlas", universe[tenth // 2 :], groups, samples, cells
    )
    collection = CubeCollection(
        cubes,
        align_groups_by="cell_ontology_id",
        group_mapping=MAPPING,
        within_study=["donor"],
    )
    return cubes, collection


def run_references(directory, cubes):
    spec = {
        "studies": list(cubes),
        "mapping": MAPPING,
        "within_study": ["donor"],
        "analyses": [
            {"name": name, "studies": [name], "formula": FORMULAS[name]}
            for name in cubes
        ]
        + [
            {
                "name": "joint",
                "studies": ["lupus", "covid"],
                "formula": FORMULAS["lupus"],
            }
        ],
    }
    samples = {}
    for name, cube in cubes.items():
        to_anndata(cube).write_h5ad(directory / f"{name}.h5ad")
        samples[name] = {
            column: {"dtype": str(values.dtype), "values": values.tolist()}
            for column, values in cube.samples.to_dict().items()
        }
        folder = directory / "r" / name
        folder.mkdir(parents=True)
        groups = cube.groups
        ontology = groups["cell_ontology_id"] if "cell_ontology_id" in groups else None
        texts = {
            "genes.txt": cube.features["feature_id"].tolist(),
            "groups.txt": groups["group_id"].tolist(),
            "ontology.txt": ["NA"] * cube.n_groups if ontology is None else ontology,
            "measured.txt": [str(bool(m)).upper() for m in cube.features["measured"]],
        }
        for file, lines in texts.items():
            (folder / file).write_text("\n".join(lines) + "\n")
        table = cube.samples
        rows = zip(table["sample_id"], table["donor"], table["condition"], strict=True)
        lines = ["sample_id\tdonor\tcondition"] + ["\t".join(map(str, r)) for r in rows]
        (folder / "samples.tsv").write_text("\n".join(lines) + "\n")
        (folder / "counts.bin").write_bytes(
            cube.layer("counts").astype("<f8").tobytes()
        )
        (folder / "n_cells.bin").write_bytes(cube.n_cells.T.astype("<f8").tobytes())
    (directory / "spec.json").write_text(json.dumps(spec))
    (directory / "samples.json").write_text(json.dumps(samples))
    (directory / "reference.py").write_text(REFERENCE_PROGRAM)
    (directory / "compare.R").write_text(R_SCRIPT)
    python = os.environ.get("REFERENCE_PYTHON", sys.executable)
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(
        [python, "-I", str(directory / "reference.py"), str(directory)],
        check=True,
        env=environment,
    )
    subprocess.run(
        [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", "compare.R", "."],
        check=True,
        cwd=directory,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY)),
    )
    reference = dict(np.load(directory / "reference.npz"))
    reference |= json.loads((directory / "reference.json").read_text())
    limma = {}
    for folder in (directory / "r").iterdir():
        if "__" in folder.name:
            genes = (folder / "genes.txt").read_text().split()
            values = np.fromfile(folder / "table.bin", dtype="<f8").reshape(7, -1)
            limma[folder.name] = (genes, values)
    reference["limma"] = limma
    versions = (directory / "versions.txt").read_text().split("\n")[:3]
    return reference, versions


CHECKS = {
    "samples": ("samples: ids, study, columns and values", "pandas concat", "mismatches"),
    "genes": ("features: union, in order", "anndata outer join, pandas.unique", "same"),
    "measured": ("measured: which study measured which gene", "anndata outer join, NaN where not measured", "mismatches"),
    "aligned": ("aligned cubes: every layer of every study", "anndata outer join", "max rel diff"),
    "axes": ("stack: genes in order, groups, samples", "anndata inner join", "same"),
    "stack": ("stack: every layer", "anndata inner join", "max rel diff"),
    "cells": ("stack: cells per entry; entries without cells", "anndata inner join n_cells; no row", "max abs diff"),
    "limma": ("limma_voom on its aligned cube", "limma, study by study", "max rel diff"),
    "status": ("tested, not tested, not measured", "limma's tables, measured genes", "mismatches"),
    "joint": ("joint model of lupus and covid, donors study::donor", "limma on the stacked counts", "max rel diff"),
    "control": ("control: the same with the donor ids of the studies", "limma on the stacked counts", "max rel diff of logFC"),
}  # fmt: skip


def difference(ours, theirs, p_value=False):
    """The largest |ours - theirs| / scale, as the module docstring describes."""
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    if ours.shape != theirs.shape:
        return np.inf
    if p_value:
        scale = np.maximum(theirs, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(theirs), 1.0)
    return float(np.max(np.abs(ours - theirs) / scale, initial=0.0))


def entries(cube, groups, samples):
    """The group and sample positions in a cube of a join's rows."""
    group = {g: k for k, g in enumerate(cube.groups["group_id"].tolist())}
    sample = {s: k for k, s in enumerate(cube.samples["sample_id"].tolist())}
    return [group[g] for g in groups], [sample[s] for s in samples]


def check_alignment(collection, reference):
    """Rows comparing the samples, genes, mask and joins with anndata and pandas."""
    samples = reference["samples"]
    wrong = list(collection.samples.columns) != ["sample_id", "study"] + [
        c for c in samples if c not in ("sample_id", "study")
    ]
    for column, theirs in samples.items():
        for a, b in zip(collection.samples[column].tolist(), theirs["values"]):
            missing = a is None or (isinstance(a, float) and np.isnan(a))
            wrong += (b is None) != missing or (b is not None and a != b)
    rows = [(*CHECKS["samples"], int(wrong), 0)]
    genes = collection.features["feature_id"].tolist()
    union = reference["outer_genes"].tolist()
    order = genes == reference["first_listed"].tolist() and set(genes) == set(union)
    rows.append((*CHECKS["genes"], order, True))
    where = {gene: k for k, gene in enumerate(union)}
    study = reference["outer_study"]
    outer = reference["outer_counts"][:, [where[gene] for gene in genes]]
    present = [~np.isnan(outer[study == s]).all(axis=0) for s in collection.studies]
    mismatches = int((np.array(present) != collection.measured).sum())
    rows.append((*CHECKS["measured"], mismatches, 0))
    largest = 0.0
    for name in collection.studies:
        cube, mine = collection.aligned_cube(name), study == name
        groups, samples = entries(
            cube, reference["outer_group"][mine], reference["outer_sample_id"][mine]
        )
        own = [where[gene] for gene in cube.features["feature_id"].tolist()]
        for layer in LAYERS:
            joined = reference[f"outer_{layer}"][mine][:, own]
            largest = max(
                largest, difference(cube.layer(layer)[groups, samples], joined)
            )
    rows.append((*CHECKS["aligned"], largest, EXACT))
    stacked = collection.stack()
    same = (
        stacked.features["feature_id"].tolist() == reference["inner_genes"].tolist()
        and set(stacked.groups["group_id"]) == set(reference["inner_group"])
        and stacked.samples.equals(collection.samples)
    )
    rows.append((*CHECKS["axes"], same, True))
    keys = zip(reference["inner_study"], reference["inner_sample_id"], strict=True)
    groups, samples = entries(
        stacked, reference["inner_group"], [f"{s}::{i}" for s, i in keys]
    )
    largest = max(
        difference(stacked.layer(layer)[groups, samples], reference[f"inner_{layer}"])
        for layer in LAYERS
    )
    rows.append((*CHECKS["stack"], largest, EXACT))
    cells = stacked.n_cells.copy()
    joined = np.abs(cells[samples, groups] - reference["inner_n_cells"]).max()
    cells[samples, groups] = 0  # entries without a row in the join have no cells
    empty = np.ones(stacked.layer("counts").shape[:2], bool)
    empty[groups, samples] = False
    zeros = stacked.layer("counts")[empty].max(initial=0)
    rows.append((*CHECKS["cells"], float(max(joined, cells.max(), zeros)), 0))
    return rows


def check_limma(collection, reference):
    """Rows comparing limma_voom on every study and on the stack with limma."""
    contrast = {"stim": "conditionstim"}
    levels = {"condition": ["ctrl", "stim"]}
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        design = Design(cube.samples, FORMULAS[study], levels=levels)
        results[study] = limma_voom(cube, design, contrasts=contrast)
    tables = {
        group: align_study_results(collection, results, group, "stim")
        for group in collection.groups["group_id"].tolist()
    }
    rows = []
    for row, study in enumerate(collection.studies):
        largest, wrong, fitted = 0.0, 0, 0
        for group, table in tables.items():
            mine = table.filter(table["study"] == study)
            found = reference["limma"].get(f"{study}__{group.replace(':', '_')}")
            if found is None:  # the study does not have the group
                wrong += int(np.isin(mine["status"], ["tested", "not_tested"]).sum())
                continue
            fitted += 1
            genes, values = found
            where = {gene: k for k, gene in enumerate(mine["feature_id"].tolist())}
            positions = [where.get(gene, -1) for gene in genes]
            expected = np.where(collection.measured[row], "not_tested", "not_measured")
            expected = expected.astype(object)
            expected[[p for p in positions if p >= 0]] = "tested"
            wrong += int((mine["status"] != expected).sum()) + positions.count(-1)
            for k, column in enumerate(COLUMNS):
                ours = mine[column][positions]
                largest = max(largest, difference(ours, values[k], "P" in TABLE[k]))
        check, *rest = CHECKS["limma"]
        rows.append((f"{study}: {check}, {fitted} cell types", *rest, largest, EXACT))
        check, *rest = CHECKS["status"]
        rows.append((f"{study}: {check}", *rest, wrong, 0))
    joint = collection.select(studies=["lupus", "covid"]).stack()
    own_donors = [d.split("::")[1] for d in joint.samples["donor"]]
    for key, samples in (
        ("joint", joint.samples),
        ("control", joint.samples.with_columns(donor=own_donors)),
    ):
        design = Design(samples, FORMULAS["lupus"], levels=levels)
        result = limma_voom(joint, design, contrasts=contrast)
        # The control compares log2 fold changes, whose difference reads plainly,
        # over the genes both tested: its design filters other genes.
        columns = COLUMNS[:1] if key == "control" else COLUMNS
        largest, other_genes = 0.0, 0
        for group in joint.groups["group_id"].tolist():
            genes, values = reference["limma"][f"joint__{group.replace(':', '_')}"]
            table = result.tables[(group, "stim")]
            ours = {gene: k for k, gene in enumerate(table["feature_id"].tolist())}
            both = [k for k, gene in enumerate(genes) if gene in ours]
            other_genes += len(genes) + len(ours) - 2 * len(both)
            for k, column in enumerate(columns):
                mine = table[column][[ours[genes[j]] for j in both]]
                p_value = "P" in TABLE[k]
                largest = max(largest, difference(mine, values[k][both], p_value))
        if key == "joint" and other_genes:
            largest = np.inf  # the genes tested must be the same
        rows.append((*CHECKS[key], largest, EXACT if key == "joint" else "control"))
    return rows


def print_table(rows):
    print("| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    passed = True
    for check, reference, metric, value, tolerance in rows:
        shown = f"{value:.2g}" if isinstance(value, float) else str(value)
        if tolerance == "control":  # must exceed the exact tier
            ok, limit = bool(value > EXACT), f"> {EXACT:.0e}"
        elif isinstance(tolerance, bool):
            ok, limit = bool(value == tolerance), str(tolerance)
        else:
            ok, limit = bool(value <= tolerance), f"{tolerance:.2g}"
        result = ("control" if tolerance == "control" else "pass") if ok else "FAIL"
        passed &= ok
        print(f"| {check} | {reference} | {metric} | {shown} | {limit} | {result} |")
    return passed


def write_fixtures(directory):
    """Small studies, and what anndata, pandas and limma make of them, for tests:
    the integer layers and cell numbers, with counts -1 where a study lacks a gene
    (NaN in the outer join)."""
    cubes, _ = build(SEED + 1, (40, 14))
    reference, _ = run_references(directory, cubes)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, cube in cubes.items():
        for layer in ("counts", "detected"):
            arrays[f"{name}/{layer}"] = cube.layer(layer)
        arrays[f"{name}/n_cells"] = cube.n_cells
    outer = reference["outer_counts"]
    arrays["outer_counts"] = np.where(np.isnan(outer), -1, outer).astype(np.int64)
    for key in ("first_listed", "outer_genes", "outer_study", "outer_sample_id"):
        arrays[key] = reference[key]
    arrays["outer_group"] = reference["outer_group"]
    for key in ("genes", "study", "sample_id", "group", "counts", "detected"):
        arrays[f"inner_{key}"] = reference[f"inner_{key}"]
    arrays["inner_n_cells"] = reference["inner_n_cells"].astype(np.int64)
    for key in ("counts", "detected"):
        arrays[f"inner_{key}"] = arrays[f"inner_{key}"].astype(np.int64)
    for key, (genes, values) in reference["limma"].items():
        arrays[f"limma/{key}/genes"] = np.array(genes)
        arrays[f"limma/{key}/values"] = values
    np.savez_compressed(FIXTURE_DIR / "studies.npz", **arrays)
    tables = {
        name: {
            "samples": {c: v.tolist() for c, v in cube.samples.to_dict().items()},
            "groups": {c: v.tolist() for c, v in cube.groups.to_dict().items()},
            "features": {c: v.tolist() for c, v in cube.features.to_dict().items()},
        }
        for name, cube in cubes.items()
    }
    record = {"studies": tables, "samples": reference["samples"], "mapping": MAPPING}
    (FIXTURE_DIR / "studies.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {FIXTURE_DIR / 'studies.npz'} and {FIXTURE_DIR / 'studies.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        cubes, collection = build(SEED, (1000, 30))
        reference, r_versions = run_references(Path(temporary), cubes)
        if arguments.write_fixtures:
            with tempfile.TemporaryDirectory() as small:
                write_fixtures(Path(small))
    versions = ", ".join(f"{k} {v}" for k, v in reference["versions"].items())
    print(f"Reference versions: {versions}; {', '.join(r_versions)}")
    print(f"CytePool side: numpy {np.__version__}")
    print(repr(collection))
    print()
    rows = check_alignment(collection, reference) + check_limma(collection, reference)
    return 0 if print_table(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
