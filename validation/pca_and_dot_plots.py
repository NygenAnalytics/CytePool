"""Check principal components and their figure against DESeq2, and dot plots against
scanpy.

``principal_components`` and ``plot_pca`` are compared with DESeq2's
``plotPCA(returnData = TRUE)`` and the ggplot it draws, on the same values: 12 samples'
log2 counts per million (three conditions, two batches, negative binomial counts of 300
genes, seed 20261103), with the 100 genes of largest variance (components 1 and 2, and
3 and 4) and with 500, more than there are; a matrix whose genes 19 to 22 have equal
variance, with the 20 of largest variance, which keeps the first of them; and the
samples of a cube's group, two of which have fewer than 10 cells, which R prepares as
the edgeR User's Guide does (``filterByExpr``, ``normLibSizes``, ``cpm(log = TRUE,
prior.count = 2)``). Coordinates are compared up to the sign of each component, which
R's ``prcomp`` leaves arbitrary; the shares of variance, the axis labels, the equal
scales of the axes and the samples' groups of colour must agree. With two samples
DESeq2 also returns a second component, of rounding noise, which CytePool leaves out.

``plot_dotplot`` is compared with ``scanpy.pl.dotplot`` on the cells that the cube was
pooled from: 4 cell types in 6 samples (one sample without cells of one type, seed
20261104), 40 genes, the cells normalized by scanpy (``normalize_total`` and ``log1p``)
and the cube by ``from_cells``. The areas of the dots and the fractions they encode, the
values their colours show, plain, scaled by gene or by group, and over the cells that
express a gene only, for the cells of every sample and for three samples and three cell
types in a chosen order, and the brackets of genes given in groups and the key of the
areas, must agree.

Tiers: exact (1e-10, relative to max(1, abs(reference))) for values both programs
compute by the same deterministic steps; texts and groupings must be identical.

Requirements:
    Python 3.13 with numpy, numba and matplotlib, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with DESeq2, edgeR, ggplot2 and jsonlite (``RSCRIPT``, default
    ``Rscript``); R runs with 8 GB of address space at most.
    Python with scanpy 1.12 (``SCANPY_PYTHON``, default this interpreter).

Usage:
    PYTHONPATH=. python validation/pca_and_dot_plots.py
    PYTHONPATH=. python validation/pca_and_dot_plots.py --write-fixtures
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
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from cytepool import plotting
from cytepool.cube import Cube, from_cells
from cytepool.normalization import log_cpm, tmm_factors
from cytepool.principal_components import principal_components

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "pca_and_dot_plots"
)
EXACT = 1e-10
# scanpy's dot plots: the area of the largest dot and the power of the fractions.
LARGEST_DOT, SIZE_EXPONENT = 200.0, 1.5
MEMORY = 8_000_000_000  # bytes of address space for R
PCA_SEED, DOT_SEED = 20261103, 20261104
CONDITIONS = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
BATCHES = ["x", "y"] * 6
SAMPLES = [f"S{i}" for i in range(1, 13)]
PCA_CASES = {"top100": (100, (1, 2)), "top100_pc34": (100, (3, 4))}
PCA_CASES["top500"] = (500, (1, 2))
GROUPS = ["B", "T", "NK", "pDC"]
CELL_SAMPLES = [f"D{i}" for i in range(1, 7)]
MARKERS = {"B cells": ["g00", "g01", "g02"], "T cells": ["g03", "g04"]}
MARKERS["others"] = ["g10", "g20", "g30", "g39"]
GENES = [gene for genes in MARKERS.values() for gene in genes]
DOT_CASES: dict[str, dict[str, Any]] = {
    "plain": {},
    "scaled by gene": {"standard_scale": "gene"},
    "scaled by group": {"standard_scale": "group"},
    "expressing cells": {"mean_only_expressed": True},
    "chosen samples and groups": {
        "samples": ["D2", "D3", "D5"],
        "groups": ["NK", "B", "T"],
    },
    "genes in groups": {"genes": MARKERS},
}

R_PROGRAM = r"""
suppressPackageStartupMessages({
  library(DESeq2); library(SummarizedExperiment); library(edgeR); library(ggplot2)
})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(name) readBin(file.path(root, name), "double",
                             n = file.size(file.path(root, name)) / 8, size = 8)
samples <- jsonlite::read_json(file.path(root, "samples.json"), simplifyVector = TRUE)
info <- data.frame(condition = factor(samples$condition), batch = factor(samples$batch),
                   row.names = samples$sample_id)
transformed <- function(values, rows = seq_len(nrow(info))) {
  colnames(values) <- rownames(info)[rows]
  DESeqTransform(SummarizedExperiment(list(values), colData = info[rows, , drop = FALSE]))
}
pca <- function(object, ntop, pcs = 1:2) {
  d <- suppressMessages(plotPCA(object, intgroup = c("condition", "batch"),
                                ntop = ntop, returnData = TRUE, pcsToUse = pcs))
  list(x = d[[1]], y = d[[2]], percent = attr(d, "percentVar"),
       name = as.character(d$name), group = as.character(d$group))
}
values <- matrix(rd("values.bin"), ncol = 12)  # genes x samples
out <- list()
object <- transformed(values)
out$top100 <- pca(object, 100)
out$top100_pc34 <- pca(object, 100, 3:4)
out$top500 <- pca(object, 500)
ties <- matrix(rd("ties.bin"), ncol = 12)
out$ties <- pca(transformed(ties), 20)
out$two_samples <- pca(transformed(values[, 1:2], 1:2), 100)
p <- suppressWarnings(suppressMessages(
  plotPCA(object, intgroup = c("condition", "batch"), ntop = 100)))
layer <- ggplot_build(p)$data[[1]]
out$figure <- list(x = layer$x, y = layer$y, colour = layer$colour,
                   xlabel = p$labels$x, ylabel = p$labels$y,
                   ratio = p$coordinates$ratio)
counts <- matrix(rd("counts.bin"), ncol = 12)
used <- as.logical(rd("used.bin"))
y <- DGEList(counts[, used])
y <- normLibSizes(y[suppressWarnings(filterByExpr(y)), , keep.lib.sizes = FALSE])
out$cube <- pca(transformed(cpm(y, log = TRUE, prior.count = 2), which(used)), 500)
out$versions <- c(paste("R", getRversion()), paste("DESeq2", packageVersion("DESeq2")),
                  paste("edgeR", packageVersion("edgeR")),
                  paste("ggplot2", packageVersion("ggplot2")))
jsonlite::write_json(out, file.path(root, "out.json"), digits = NA, auto_unbox = TRUE)
"""

SCANPY_PROGRAM = textwrap.dedent(
    '''
    """scanpy's dot plots of the cells, normalized by scanpy."""

    import json
    import sys
    import warnings
    from importlib.metadata import version

    import matplotlib

    matplotlib.use("Agg")
    import anndata
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    data = np.load(f"{directory}/cells.npz")
    cases = json.loads(open(f"{directory}/cases.json").read())
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(data["samples"]),
            "cell_type": pd.Categorical(data["groups"], categories=list(data["order"])),
        },
        index=[f"cell{i}" for i in range(data["groups"].size)],
    )
    cells = anndata.AnnData(
        X=sp.csr_matrix(data["counts"].astype(np.float64)),
        obs=obs,
        var=pd.DataFrame(index=list(data["genes"])),
    )
    sc.pp.normalize_total(cells, target_sum=1e4)
    sc.pp.log1p(cells)
    out = {}
    for name, case in cases.items():
        subset = cells
        if case["samples"] is not None:
            subset = subset[subset.obs["sample"].isin(case["samples"])]
        if case["groups"] is not None:
            subset = subset[subset.obs["cell_type"].isin(case["groups"])]
        subset = subset.copy()
        for column in ("sample", "cell_type"):
            subset.obs[column] = subset.obs[column].cat.remove_unused_categories()
        plot = sc.pl.dotplot(
            subset,
            case["genes"],
            groupby="cell_type",
            categories_order=case["groups"],
            standard_scale={"gene": "var", "group": "group", None: None}[
                case["standard_scale"]
            ],
            mean_only_expressed=case["mean_only_expressed"],
            return_fig=True,
            show=False,
        )
        plot.make_figure()
        legend = plot.ax_dict["size_legend_ax"].get_xticklabels()
        groups = plot.var_groups
        out[name] = {
            "rows": [str(v) for v in plot.dot_color_df.index],
            "columns": [str(v) for v in plot.dot_color_df.columns],
            "color": plot.dot_color_df.to_numpy().tolist(),
            "fraction": plot.dot_size_df.to_numpy().tolist(),
            "sizes": plot.ax_dict["mainplot_ax"].collections[0].get_sizes().tolist(),
            "key": [text.get_text() for text in legend],
            "gene_groups": None
            if groups is None
            else [list(groups.labels), [list(p) for p in groups.positions]],
        }
    out["versions"] = {
        package: version(package) for package in ("scanpy", "anndata", "matplotlib")
    }
    with open(f"{directory}/out.json", "w") as handle:
        json.dump(out, handle)
    '''
)


# Data.


def pca_data() -> dict[str, np.ndarray]:
    """Counts of 12 samples, their log-CPM values, a matrix with tied variances, and
    the cell numbers of a cube of the counts."""
    rng = np.random.default_rng(PCA_SEED)
    n_genes = 300
    means = np.exp(rng.normal(4.0, 1.5, n_genes))
    effects = np.zeros((12, n_genes))
    for k, level in enumerate("abc"):
        changed = rng.choice(n_genes, 40, replace=False)
        rows = [i for i, value in enumerate(CONDITIONS) if value == level]
        effects[np.ix_(rows, changed)] = rng.normal(0.0, 1.0 + 0.3 * k, 40)
    effects[1::2, rng.choice(n_genes, 30, replace=False)] += 0.7  # batch y
    mu = means * np.exp(effects) * rng.uniform(0.6, 1.6, (12, 1))
    counts = rng.negative_binomial(8.0, 8.0 / (8.0 + mu)).astype(np.float64)
    values = log_cpm(counts, norm_factors=tmm_factors(counts))
    # 18 genes of distinct, large variance, then 4 of variance 12 (permutations of
    # one profile, whose variances are equal to the last bit), then 6 small ones.
    base = np.array([0.0, 1.0, 3.0, 4.0, 7.0, 9.0] * 2)
    ties = np.vstack(
        [rng.normal(0.0, 10.0 + j, 12) for j in range(18)]  # fmt: skip
        + [base[order] for order in (np.arange(12), np.arange(12)[::-1])]
        + [base[np.roll(np.arange(12), shift)] for shift in (1, 5)]
        + [rng.normal(0.0, 0.1, 12) for _ in range(6)]
    ).T  # samples x genes
    n_cells = np.where(np.arange(12) < 10, 60, 5)
    return {"counts": counts, "values": values, "ties": ties, "n_cells": n_cells}


def dot_data() -> dict[str, np.ndarray]:
    """Cells of 4 types in 6 samples with their counts of 40 genes; sample D6 has no
    pDC."""
    rng = np.random.default_rng(DOT_SEED)
    n_genes = 40
    sizes = rng.integers(15, 120, (len(CELL_SAMPLES), len(GROUPS)))
    sizes[-1, -1] = 0
    samples, groups = [], []
    for i, sample in enumerate(CELL_SAMPLES):
        for g, group in enumerate(GROUPS):
            samples += [sample] * int(sizes[i, g])
            groups += [group] * int(sizes[i, g])
    n_cells = len(groups)
    base = np.exp(rng.normal(-1.0, 1.2, n_genes))
    level = np.ones((len(GROUPS), n_genes))
    for g in range(len(GROUPS)):  # markers: genes 0-2 in B, 3-4 in T, and so on
        level[g, rng.choice(n_genes, 6, replace=False)] *= rng.uniform(3.0, 12.0, 6)
    level[0, :3] *= 20.0
    level[1, 3:5] *= 20.0
    code = np.array([GROUPS.index(group) for group in groups])
    depth = rng.uniform(0.5, 2.0, n_cells)[:, np.newaxis]
    mu = base * level[code] * depth
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(np.float64)
    counts[counts.sum(axis=1) == 0, 39] = 1.0  # every cell has counts
    return {
        "counts": counts,
        "samples": np.array(samples),
        "groups": np.array(groups),
        "order": np.array(GROUPS),
        "genes": np.array([f"g{j:02d}" for j in range(n_genes)]),
    }


def dot_cube(data: dict[str, np.ndarray]) -> Cube:
    return from_cells(
        data["counts"],
        samples=data["samples"],
        groups=data["groups"],
        sample_table={"sample_id": CELL_SAMPLES},
        group_table={"group_id": GROUPS},
        features=data["genes"].tolist(),
        layers=["counts", "detected", "log_sum"],
    )


def pca_cube(data: dict[str, np.ndarray]) -> Cube:
    return Cube(
        {"counts": data["counts"][np.newaxis].astype(np.uint32)},
        samples={"sample_id": SAMPLES, "condition": CONDITIONS, "batch": BATCHES},
        groups={"group_id": ["T"]},
        features={"feature_id": [f"g{j:03d}" for j in range(data["counts"].shape[1])]},
        n_cells=data["n_cells"][:, np.newaxis],
    )


# The references, each run as a separate program.


def run_r(data: dict[str, np.ndarray]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in ("values", "ties", "counts"):
            (root / f"{name}.bin").write_bytes(
                np.ascontiguousarray(data[name], dtype="<f8").tobytes()
            )
        used = (data["n_cells"] >= 10).astype(np.float64)
        (root / "used.bin").write_bytes(used.astype("<f8").tobytes())
        table = {"sample_id": SAMPLES, "condition": CONDITIONS, "batch": BATCHES}
        (root / "samples.json").write_text(json.dumps(table))
        (root / "program.R").write_text(R_PROGRAM)
        limit = (MEMORY, MEMORY)
        subprocess.run(
            [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", "program.R", "."],
            check=True,
            cwd=root,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, limit),
        )
        result: dict[str, Any] = json.loads((root / "out.json").read_text())
        return result


def dot_cases() -> dict[str, dict[str, Any]]:
    """Every case with each option, its default where the case does not set it."""
    defaults = {
        "genes": GENES,
        "samples": None,
        "groups": None,
        "standard_scale": None,
        "mean_only_expressed": False,
    }
    return {name: defaults | case for name, case in DOT_CASES.items()}


def run_scanpy(data: dict[str, np.ndarray]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        np.savez(root / "cells.npz", **data)
        (root / "cases.json").write_text(json.dumps(dot_cases()))
        (root / "program.py").write_text(SCANPY_PROGRAM)
        python = os.environ.get("SCANPY_PYTHON", sys.executable)
        subprocess.run([python, "-I", "program.py", str(root)], check=True, cwd=root)
        result: dict[str, Any] = json.loads((root / "out.json").read_text())
        return result


# CytePool's figures, and the comparisons.


def axes() -> Any:
    return Figure().subplots()


def artists(ax: Any, gid: str) -> list[Any]:
    return [a for a in ax.get_children() if a.get_gid() == gid]


def scaled(ours: Any, reference: Any) -> float:
    """Largest |ours - reference| over max(|reference|, 1); inf for unequal shapes."""
    ours, reference = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if ours.shape != reference.shape:
        return np.inf
    if ours.size == 0:
        return 0.0
    return float(np.max(np.abs(ours - reference) / np.maximum(np.abs(reference), 1.0)))


def up_to_sign(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, int]:
    """The difference of two coordinate matrices after each column of ours takes the
    sign that matches theirs, and the number of columns whose sign differed."""
    signs = np.sign(np.sum(ours * theirs, axis=0))
    return scaled(ours * signs, theirs), int(np.sum(signs < 0))


def figure_samples(ax: Any) -> tuple[np.ndarray, list[tuple[float, ...]]]:
    """The drawn coordinates of the samples, in their order (that of their labels),
    and the colour of each."""
    positions = np.array([label.xy for label in artists(ax, "sample labels")])
    colours: list[tuple[float, ...]] = [()] * len(positions)
    for collection in artists(ax, "samples"):
        faces = collection.get_facecolors()
        for offset, face in zip(collection.get_offsets(), faces, strict=True):
            i = int(np.argmin(np.abs(positions - offset).sum(axis=1)))
            colours[i] = tuple(face.tolist())
    return positions, colours


def partition(labels: list[Any]) -> set[frozenset[int]]:
    """The samples grouped by their labels, whatever the labels are."""
    groups: dict[Any, set[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, set()).add(i)
    return {frozenset(members) for members in groups.values()}


def row(check: str, reference: str, value: float, tolerance: float) -> tuple:
    return (
        check,
        reference,
        "max scaled difference",
        f"{value:.1e}",
        f"{tolerance:.0e}",
        "pass" if value <= tolerance else "FAIL",
    )


def pca_rows(data: dict[str, np.ndarray], r: dict[str, Any]) -> list[tuple]:
    rows = []
    flipped = 0
    worst = 0.0
    for case, (top, (first, second)) in PCA_CASES.items():
        result = principal_components(data["values"], top=top)
        ours = result.coordinates[:, [first - 1, second - 1]]
        theirs = np.column_stack([r[case]["x"], r[case]["y"]])
        difference, flips = up_to_sign(ours, theirs)
        shares = result.variance_explained[[first - 1, second - 1]]
        worst = max(worst, difference, scaled(shares, r[case]["percent"]))
        flipped += flips
    rows.append(
        row(
            "Log-CPM values of 12 samples, the 100 genes of largest variance "
            "(components 1 and 2, 3 and 4) and 500, more than the 300 genes: "
            "coordinates up to sign, shares of variance",
            "plotPCA(returnData = TRUE)",
            worst,
            EXACT,
        )
    )
    result = principal_components(data["ties"], top=20)
    theirs = np.column_stack([r["ties"]["x"], r["ties"]["y"]])
    difference, flips = up_to_sign(result.coordinates[:, :2], theirs)
    flipped += flips
    keeps_first = result.genes.tolist()[-2:] == [18, 19]
    rows.append(
        row(
            "Genes 19 to 22 of equal variance, the 20 of largest variance: the first "
            "two of them kept; coordinates up to sign",
            "plotPCA(returnData = TRUE)",
            difference if keeps_first else np.inf,
            EXACT,
        )
    )
    cube = pca_cube(data)
    result = principal_components(cube)
    theirs = np.column_stack([r["cube"]["x"], r["cube"]["y"]])
    difference, flips = up_to_sign(result.coordinates[:, :2], theirs)
    flipped += flips
    same_samples = result.samples["sample_id"].tolist() == r["cube"]["name"]
    rows.append(
        row(
            "A cube's group: the 10 samples with at least 10 cells, filterByExpr, TMM, "
            "log-CPM with an average count of 2; coordinates up to sign, the samples",
            "edgeR, plotPCA(returnData = TRUE)",
            difference if same_samples else np.inf,
            EXACT,
        )
    )
    figure = r["figure"]
    groups = [f"{c}:{b}" for c, b in zip(CONDITIONS, BATCHES, strict=True)]
    ax = plotting.plot_pca(
        data["values"], top=100, color_by=groups, labels=True, ax=axes()
    )
    positions, colours = figure_samples(ax)
    theirs = np.column_stack([figure["x"], figure["y"]])
    difference, _ = up_to_sign(positions, theirs)
    labels = (ax.get_xlabel(), ax.get_ylabel()) == (figure["xlabel"], figure["ylabel"])
    equal_scales = ax.get_aspect() == float(figure["ratio"])
    same_groups = (
        partition(colours)
        == partition(figure["colour"])
        == partition(r["top100"]["group"])
    )
    rows.append(
        row(
            "The figure: the points up to sign, the axis labels, equal scales, and the "
            "samples' colours by condition and batch",
            "plotPCA",
            difference if (labels and equal_scales and same_groups) else np.inf,
            EXACT,
        )
    )
    two = principal_components(data["values"][:2], top=100)
    pc1, _ = up_to_sign(
        two.coordinates[:, :1], np.array(r["two_samples"]["x"])[:, None]
    )
    noise = float(np.asarray(r["two_samples"]["percent"])[1])
    rows.append(
        (
            "Two samples: PC1; DESeq2 returns a second component of rounding noise "
            f"(share {noise:.1e}), CytePool one component",
            "plotPCA(returnData = TRUE)",
            "max scaled difference of PC1",
            f"{pc1:.1e}",
            "n/a",
            "deliberate difference"
            if two.n_components == 1 and pc1 <= EXACT
            else "FAIL",
        )
    )
    rows.append(
        (
            "Components whose sign differs from R's (prcomp's signs are arbitrary)",
            "plotPCA(returnData = TRUE)",
            "count",
            str(flipped),
            "n/a",
            "information",
        )
    )
    return rows


def dot_rows(cube: Cube, scanpy: dict[str, Any]) -> list[tuple]:
    rows = []
    for name, case in dot_cases().items():
        theirs = scanpy[name]
        ax = plotting.plot_dotplot(
            cube,
            case["genes"],
            groups=case["groups"],
            samples=case["samples"],
            standard_scale=case["standard_scale"],
            mean_only_expressed=case["mean_only_expressed"],
            ax=axes(),
        )
        dots = artists(ax, "dots")[0]
        sizes = dots.get_sizes()
        fraction = np.array(theirs["fraction"])
        # The fractions that the areas encode, with scanpy's largest fraction shown.
        f_max = np.ceil(fraction.max() * 10) / 10
        encoded = f_max * (sizes / LARGEST_DOT) ** (1.0 / SIZE_EXPONENT)
        difference = max(
            scaled(sizes, theirs["sizes"]),
            scaled(encoded, fraction.ravel()),
            scaled(np.asarray(dots.get_array()), np.ravel(theirs["color"])),
        )
        names = (
            [t.get_text() for t in ax.get_yticklabels()] == theirs["rows"]
            and [t.get_text() for t in ax.get_xticklabels()] == theirs["columns"]
            and key_labels(ax) == theirs["key"]
        )
        what = {
            "plain": "fractions and means of 9 genes in 4 cell types, pooled over 6 "
            "samples",
            "scaled by gene": "means scaled by gene (standard_scale = 'var')",
            "scaled by group": "means scaled by group (standard_scale = 'group')",
            "expressing cells": "means over the cells that express the gene "
            "(mean_only_expressed)",
            "chosen samples and groups": "3 samples and 3 cell types, in a chosen order",
            "genes in groups": "genes given in 3 groups",
        }[name]
        rows.append(
            row(
                f"Dot plot, {what}: the areas, the fractions they encode, the colour "
                "values, the names of the rows and columns, the key of the areas",
                "scanpy.pl.dotplot",
                difference if names else np.inf,
                EXACT,
            )
        )
        if theirs["gene_groups"] is not None:
            labels, positions = theirs["gene_groups"]
            brackets = [line.get_xdata() for line in artists(ax, "gene groups")]
            ours = [[round(x[0] + 0.3), round(x[-1] - 0.3)] for x in brackets]
            texts = [t.get_text() for t in artists(ax, "gene group names")]
            same = ours == positions and texts == labels
            rows.append(
                (
                    "Dot plot, genes given in 3 groups: the columns of each bracket "
                    "and its name",
                    "scanpy.pl.dotplot",
                    "identical",
                    "yes" if same else "no",
                    "0",
                    "pass" if same else "FAIL",
                )
            )
    return rows


def key_labels(ax: Any) -> list[str]:
    """The fractions of the key of a dot plot's areas, as its legend writes them."""
    legends = [a.get_legend() for a in ax.figure.axes if a.get_legend() is not None]
    return [text.get_text() for text in legends[0].get_texts()]


def write_fixtures(
    pca: dict[str, np.ndarray],
    r: dict[str, Any],
    cube: Cube,
    scanpy: dict[str, Any],
) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    arrays = {
        "values": pca["values"],
        "ties": pca["ties"],
        "counts": pca["counts"].astype(np.int32),
        "n_cells": pca["n_cells"],
    }
    for case in (*PCA_CASES, "ties", "cube"):
        arrays[f"{case} coordinates"] = np.column_stack([r[case]["x"], r[case]["y"]])
        arrays[f"{case} percent"] = np.array(r[case]["percent"], dtype=float)
    figure = r["figure"]
    arrays["figure coordinates"] = np.column_stack([figure["x"], figure["y"]])
    arrays["figure labels"] = np.array([figure["xlabel"], figure["ylabel"]])
    np.savez_compressed(FIXTURE_DIRECTORY / "pca.npz", **arrays)
    arrays = {name: np.asarray(cube.layer(name)) for name in ("detected", "log_sum")}
    arrays["n_cells"] = np.asarray(cube.n_cells)
    for name in DOT_CASES:
        for field in ("color", "fraction", "sizes"):
            arrays[f"{name} {field}"] = np.array(scanpy[name][field], dtype=float)
    np.savez_compressed(FIXTURE_DIRECTORY / "dot_plots.npz", **arrays)
    texts = {
        name: {k: scanpy[name][k] for k in ("rows", "columns", "key", "gene_groups")}
        for name in DOT_CASES
    }
    (FIXTURE_DIRECTORY / "dot_plots.json").write_text(json.dumps(texts, indent=1))
    print(f"wrote {FIXTURE_DIRECTORY}/pca.npz, dot_plots.npz and dot_plots.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    args = parser.parse_args()
    pca = pca_data()
    r = run_r(pca)
    cells = dot_data()
    cube = dot_cube(cells)
    scanpy = run_scanpy(cells)
    versions = list(r["versions"]) + [f"{k} {v}" for k, v in scanpy["versions"].items()]
    print("Reference versions: " + ", ".join(versions) + "\n")
    rows = pca_rows(pca, r) + dot_rows(cube, scanpy)
    header = ["check", "reference", "metric", "value", "tolerance", "result"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(" --- " for _ in header) + "|")
    for entry in rows:
        print("| " + " | ".join(entry) + " |")
    failed = any(entry[-1] == "FAIL" for entry in rows)
    if args.write_fixtures:
        write_fixtures(pca, r, cube, scanpy)
    print("\nSOME CHECKS FAILED" if failed else "\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
