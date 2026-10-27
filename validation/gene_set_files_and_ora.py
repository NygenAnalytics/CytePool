"""Check gene-set files and over-representation analysis against R, limma and decoupler.

GMT files: four files of 60 sets, simulated from fixed seeds in the styles of MSigDB
(names in capitals, links as descriptions, symbols), Reactome (names with spaces and
punctuation, carriage returns before line feeds, tabs at the ends of lines),
WikiPathways (names joined by %, Entrez ids, a byte-order mark, blank lines) and GMT
exports of the Gene Ontology (descriptions "na" or empty, genes listed twice, a set
without genes, a name outside ASCII, compressed by gzip), are read by ``read_gmt`` and
by hand in R (``readLines``, fields split at tabs). Each collection is written by
``write_gmt``, and the file read again by both. decoupler's ``read_gmt`` reads the
MSigDB-style file too; it splits fields at any white space, so it reads files only
without spaces in names or empty descriptions. ``GeneSets.restrict_to`` and
``filter_by_size`` are compared with limma's ``ids2indices`` and a filter of the sizes
in R, and ``camera`` given the sets read by CytePool with limma's ``camera`` given the
sets read in R.

Over-representation: simulated differential-expression results of 4,000 genes (8% up
and 6% down at an FDR of 5%, a block of tied p-values), and sets of up to 600 genes, a
fifth of them outside the results, enriched for genes up, down or both or for none,
with two identical sets, a set listing genes twice, a set of every gene, a set outside
the universe and a set with two features of one symbol, one up and one down.
``over_representation`` of the table is compared with limma's ``kegga`` of a fit with
the same p-values and log fold changes, which selects the genes up and down itself,
and with kegga's default method given the genes of each direction and the universe: by
feature id; by a symbol that 160 features share in pairs; by symbols that 131
features lack; with a universe of 60% of the genes and of genes outside the results;
for genes without a p-value; and for a list of genes, some outside the universe or
listed twice. The p-values adjusted by ``p.adjust``, the order of ``topKEGG`` (by
p-value, ties by the number of genes, found by probing), R's ``phyper`` on a grid that
reaches 1e-300 and underflow, and what ``plot_over_representation`` draws are compared
too. Deliberate differences: sets without genes in the universe, which kegga leaves
out of its table; genes without a p-value, which kegga's method for a fit keeps in the
universe; and features without a symbol, which it leaves out of its adjustment of the
p-values.

Tier: exact (1e-10 relative; counts, names and orders identical). With
``--write-fixtures`` the same checks run on smaller inputs (6 sets per GMT file, 300
genes to match, 400 genes and 40 sets), which the tests read.

Requirements:
    Python 3.13 with numpy, numba and matplotlib, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with limma and jsonlite (``RSCRIPT``, by default ``Rscript``).
    decoupler 2.2 (``DECOUPLER_PYTHON``, by default this interpreter).

Usage:
    PYTHONPATH=. python validation/gene_set_files_and_ora.py
    PYTHONPATH=. python validation/gene_set_files_and_ora.py --write-fixtures
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.colors
import matplotlib.figure
import numpy as np

from cytepool import Table, camera, over_representation, plotting, read_gmt, write_gmt
from cytepool.stats.distributions import hypergeom_sf
from cytepool.stats.multiple_testing import adjust_p_values

FIXTURES = Path(__file__).resolve().parents[1] / "tests/data/gene_set_files_and_ora"
EXACT = 1e-10
ALPHA = 0.05
LISTS = {"up": "Up", "down": "Down", "either": "Either", None: "DE"}
GMT_FILES = ("msigdb.gmt", "reactome.gmt", "wikipathways.gmt", "gene_ontology.gmt.gz")
# The cases where kegga of a fit differs on purpose, and why.
DELIBERATE = {
    "missing_symbols": ", which adjusts the p-values of the features with a symbol only",
    "missing_p_values": ", whose universe has the genes without a p-value",
}

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(jsonlite)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
ids_of <- function(path) {
  x <- readLines(path, encoding = "UTF-8", warn = FALSE); x[x == ""] <- NA; x
}
put <- function(dir, name, x)
  writeBin(as.double(x), file.path(dir, paste0("out ", name, ".bin")), size = 8)
by_hand <- function(path) {  # fields split at tabs; "na" or "" is no description
  con <- gzfile(path, encoding = "UTF-8-BOM"); lines <- readLines(con, warn = FALSE)
  close(con)
  fields <- lapply(strsplit(lines[trimws(lines) != ""], "\t", fixed = TRUE), trimws)
  list(names = vapply(fields, function(f) f[1], ""),
       descriptions = vapply(fields, function(f)
         if (length(f) < 2 || f[2] %in% c("", "na")) NA_character_ else f[2], ""),
       genes = lapply(fields, function(f) { g <- f[-(1:2)]; unique(g[g != ""]) }))
}
files <- readLines(file.path(root, "gmt_files.txt"))
write_json(lapply(setNames(files, files), function(f) by_hand(file.path(root, f))),
           file.path(root, "out gmt.json"), auto_unbox = FALSE, na = "null")
# ids2indices, the sizes kept and camera, on the sets read by hand.
sets <- by_hand(file.path(root, "matching.gmt"))
identifiers <- readLines(file.path(root, "identifiers.txt"))
index <- ids2indices(setNames(sets$genes, sets$names), identifiers)
writeLines(names(index), file.path(root, "out index names.txt"))
put(root, "index", unlist(index)); put(root, "index sizes", lengths(index))
kept <- index[lengths(index) >= 15 & lengths(index) <= 200]
y <- matrix(readBin(file.path(root, "y.bin"), "double", n = 8 * length(identifiers)),
            ncol = 8)
tested <- camera(y, kept, cbind(1, rep(0:1, each = 4)), contrast = 2, sort = FALSE)
writeLines(rownames(tested), file.path(root, "out camera names.txt"))
put(root, "camera", tested$PValue)
# phyper over a grid of (k, K, n, N): P(X >= k).
grid <- matrix(readBin(file.path(root, "grid.bin"), "double", n = 1e6), ncol = 4)
put(root, "phyper", phyper(grid[, 1] - 1, grid[, 2], grid[, 4] - grid[, 2], grid[, 3],
                           lower.tail = FALSE))
# kegga with the sets as gene.pathway: of a fit, and of lists of genes.
save <- function(dir, result, lists, prefix) {
  writeLines(rownames(result), file.path(dir, paste0("out ", prefix, "pathways.txt")))
  put(dir, paste0(prefix, "N"), result$N)
  for (l in lists) {
    p <- result[[paste0("P.", l)]]
    put(dir, paste0(prefix, l), result[[l]]); put(dir, paste0(prefix, "P.", l), p)
    put(dir, paste0(prefix, "FDR.", l), p.adjust(p, "BH"))
    writeLines(rownames(topKEGG(result, sort = l, number = 30)),
               file.path(dir, paste0("out ", prefix, "top ", l, ".txt")))
  }
}
for (case in readLines(file.path(root, "ora_cases.txt"))) {
  dir <- file.path(root, case)
  gp <- read.delim(file.path(dir, "gene_pathway.tsv"), header = FALSE, quote = "",
                   colClasses = "character", na.strings = character(0))
  names <- data.frame(id = unique(gp[[2]]), name = unique(gp[[2]]))
  if (file.exists(file.path(dir, "p.bin"))) {
    ids <- ids_of(file.path(dir, "ids.txt")); G <- length(ids)
    fit <- eBayes(lmFit(matrix(rnorm(G * 4), G, 4), cbind(1, c(0, 0, 1, 1))))
    fit$p.value[, 2] <- readBin(file.path(dir, "p.bin"), "double", n = G)
    fit$coefficients[, 2] <- readBin(file.path(dir, "lfc.bin"), "double", n = G)
    save(dir, kegga(fit, coef = 2, geneid = ids, FDR = 0.05, gene.pathway = gp,
                    pathway.names = names), c("Up", "Down"), "fit ")
  }
  lists <- readLines(file.path(dir, "lists.txt"))
  de <- lapply(setNames(lists, lists), function(l) ids_of(file.path(dir, paste0(l, ".txt"))))
  universe <- ids_of(file.path(dir, "universe.txt"))
  save(dir, kegga(de, universe = universe, gene.pathway = gp, pathway.names = names),
       lists, "")
  u <- unique(universe[!is.na(universe)])
  put(dir, "selected", sapply(de, function(l) length(intersect(l, u))))
  put(dir, "universe", length(u))
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma"))),
           file.path(root, "versions.txt"))
"""

DECOUPLER_PROGRAM = '''
"""Reference: decoupler's read_gmt, as set name to genes."""

import json
import sys
from importlib.metadata import version

import decoupler as dc

frame = dc.pp.read_gmt(sys.argv[1])
sets = {}
for source, target in zip(frame["source"], frame["target"]):
    sets.setdefault(str(source), []).append(str(target))
with open(sys.argv[2], "w") as handle:
    json.dump({"sets": sets, "version": version("decoupler")}, handle)
'''


# Inputs.


def gmt_files(rng, n_sets):
    """Raw GMT files in the styles of four providers, as bytes by file name."""
    symbols = [
        f"{a}{b}{n}" for a in "ABCDEFGHKMNPRSTZ" for b in "ACLRTX" for n in range(40)
    ]
    entrez = [str(n) for n in rng.choice(np.arange(1, 120_000), 3000, replace=False)]
    sizes = np.exp(rng.uniform(np.log(5), np.log(400), n_sets)).astype(int)

    def genes(pool, size):
        return [str(g) for g in rng.choice(pool, size, replace=False)]

    msigdb = [
        "\t".join([f"HALLMARK_PROCESS_{i}", "https://www.gsea-msigdb.org/gsea/msigdb/"
                   f"human/geneset/HALLMARK_PROCESS_{i}", *genes(symbols, m)])
        for i, m in enumerate(sizes)
    ]  # fmt: skip
    reactome = [
        "\t".join([f"Signaling by pathway {i} (beta, gamma)", f"R-HSA-{1000 + i}",
                   *genes(symbols, m)]) + ("\t" if i % 3 == 0 else "")
        for i, m in enumerate(sizes)
    ]  # fmt: skip
    wikipathways = [
        "\t".join([f"Pathway {i}%WikiPathways_20260110%WP{200 + i}%Homo sapiens",
                   f"https://www.wikipathways.org/pathways/WP{200 + i}",
                   *genes(entrez, m)])
        for i, m in enumerate(sizes)
    ]  # fmt: skip
    ontology = []
    for i, m in enumerate(sizes):
        members = genes(symbols, m)
        name = f"regulation of process {i}%GOBP%GO:{i:07d}"
        if i == 1:
            name = "beta-catenin destruction complex (\u03b2-catenin)%GOCC%GO:0030877"
        description = ["na", "", f"term {i}"][i % 3]
        ontology.append("\t".join([name, description, *members, *members[: m // 4]]))
    ontology.append("set without genes%GOBP%GO:9999999\tna")
    return dict(
        zip(
            GMT_FILES,
            [
                ("\n".join(msigdb) + "\n").encode(),
                ("\r\n".join(reactome) + "\r\n").encode(),
                b"\xef\xbb\xbf" + ("\n\n".join(wikipathways) + "\n").encode(),
                gzip.compress(("\n".join(ontology) + "\n").encode(), mtime=0),
            ],
            strict=True,
        )
    )


def matching_inputs(rng, n_genes):
    """A GMT file of symbols, the identifiers of an analysis (some in no set, most
    sets with genes outside it) and its log-expression, 8 samples in two groups."""
    symbols = [f"SYM{i}" for i in range(int(1.3 * n_genes))]
    sizes = np.exp(rng.uniform(np.log(4), np.log(300), 60)).astype(int)
    lines = [
        "\t".join(
            [f"SET_{s}", "na", *rng.choice(symbols, size, replace=False).tolist()]
        )
        for s, size in enumerate(sizes)
    ]
    identifiers = rng.choice(symbols, n_genes, replace=False).tolist()
    y = rng.normal(8.0, 1.0, (n_genes, 8))
    y[: n_genes // 10, 4:] += 1.0
    return ("\n".join(lines) + "\n").encode(), identifiers, y


def hypergeometric_grid():
    """(k, K, n, N): small populations, the ends of the support, and deep tails."""
    rows = []
    for N in (10, 57, 1000, 20_000):
        for K in sorted({1, 2, N // 10, N // 3, N - 1, N}):
            for n in sorted({1, 3, N // 20, N // 2, N}):
                mean = K * n / N
                sd = np.sqrt(mean * (1 - K / N) * (N - n) / max(N - 1, 1))
                low, high = max(0, n - (N - K)), min(K, n)
                for k in sorted({low, 0, 1, round(mean), round(mean + 3 * sd), high}):
                    if low <= k <= high:
                        rows.append((k, K, n, N))
    rows += [(k, 500, 3000, 30_000) for k in (60, 100, 200, 300, 400, 500)]
    return np.array(rows, dtype=np.float64)


def de_results(rng, n_genes):
    """A table of results: ids, symbols (some shared by two features), log fold
    changes, p-values with a block of ties, and adjusted p-values; each gene's status;
    and a feature up and one down that share a symbol."""
    ids = np.array([f"ENSG{i:011d}" for i in range(n_genes)])
    symbols = np.array([f"S{i}" for i in range(n_genes)], dtype=object)
    shared = rng.choice(n_genes, 2 * (n_genes // 50), replace=False)
    half = len(shared) // 2
    symbols[shared[:half]] = symbols[shared[half:]]
    status = rng.choice(3, n_genes, p=[0.86, 0.08, 0.06])  # none, up, down
    changed = status > 0
    p = rng.uniform(size=n_genes)
    p[changed] = rng.beta(0.2, 30.0, changed.sum())
    p[rng.choice(np.flatnonzero(~changed), n_genes // 40, replace=False)] = 0.003
    lfc = rng.normal(0.0, 0.4, n_genes)
    lfc[status == 1] = rng.normal(1.5, 0.5, (status == 1).sum())
    lfc[status == 2] = -rng.normal(1.5, 0.5, (status == 2).sum())
    adjusted = adjust_p_values(p, "bh")
    assert np.all(np.abs(adjusted / ALPHA - 1.0) > 1e-9), "a p-value at the cutoff"
    # A symbol of a feature that goes up and of one that goes down.
    up, down = (np.flatnonzero(status == k)[np.argmin(p[status == k])] for k in (1, 2))
    symbols[down] = symbols[up]
    table = Table(
        {"feature_id": ids, "symbol": symbols, "log2_fold_change": lfc,
         "p_value": p, "adjusted_p_value": adjusted}
    )  # fmt: skip
    return table, status, (up, down)


def ora_sets(rng, ids, status, n_sets):
    """Sets of 1 to 600 genes, a fifth outside the results, enriched for genes up, down
    or both or for none; two identical, one listing genes twice, every gene, none."""
    outside = np.array([f"X{i}" for i in range(len(ids) // 4)])
    pools = {
        "none": [ids], "up": [ids, ids[status == 1]],
        "down": [ids, ids[status == 2]], "both": [ids[status == 1], ids[status == 2]],
    }  # fmt: skip
    sets = {}
    for s, size in enumerate(np.exp(rng.uniform(0.0, np.log(600), n_sets)).astype(int)):
        kind = str(rng.choice(list(pools), p=[0.6, 0.15, 0.15, 0.1]))
        drawn = [rng.choice(pool, max(1, size // 3)) for pool in pools[kind]]
        drawn.append(rng.choice(outside, max(1, size // 5)))
        sets[f"set_{s:03d}_{kind}"] = list(
            dict.fromkeys(np.concatenate(drawn).tolist())
        )
    identical = [*ids[status == 1][:12].tolist(), *ids[:30].tolist()]
    sets |= {"identical_a": identical, "identical_b": identical}
    sets["listed_twice"] = ids[100:140].tolist() * 2
    sets["every_gene"] = ids.tolist()
    sets["outside_universe"] = outside[:25].tolist()
    return sets


def selected_lists(results, column):
    """The genes up, down and in either direction, as CytePool and kegga select them:
    an adjusted p-value below ALPHA, and the sign of the log fold change."""
    ids = results[column]
    with np.errstate(invalid="ignore"):
        significant = results["adjusted_p_value"] < ALPHA
    lfc = results["log2_fold_change"]
    return {
        key: [g for g in ids[significant & rows].tolist() if g is not None]
        for key, rows in (("Up", lfc > 0), ("Down", lfc < 0), ("Either", lfc != 0))
    }


def ora_cases(rng, n_genes, n_sets):
    """Each comparison with kegga: CytePool's call; the lists and universe that kegga's
    default method gets; and, where kegga selects the genes of a fit itself, the fit."""
    table, status, pair = de_results(rng, n_genes)
    ids = table["feature_id"]
    sets = ora_sets(rng, ids, status, n_sets)
    sets["shared_symbol"] = [*ids[list(pair)].tolist(), *ids[200:210].tolist()]
    symbol_of = dict(zip(ids.tolist(), table["symbol"].tolist(), strict=True))
    by_symbol = {
        name: [symbol_of.get(g, g) for g in genes] for name, genes in sets.items()
    }
    # Features without a symbol, half of them changed, so that kegga's adjustment
    # without them moves genes across the cutoff.
    unnamed = table["symbol"].copy()
    unnamed[rng.choice(n_genes, n_genes // 60, replace=False)] = None
    unnamed[rng.choice(np.flatnonzero(status > 0), n_genes // 60, replace=False)] = None
    unnamed = table.with_columns(symbol=unnamed)
    universe = rng.choice(ids, int(0.6 * n_genes), replace=False).tolist()
    universe += [f"X{i}" for i in range(0, 200, 2)] + [f"Z{i}" for i in range(50)]
    p = np.where(rng.uniform(size=n_genes) < 0.1, np.nan, table["p_value"])
    missing = table.with_columns(p_value=p, adjusted_p_value=adjust_p_values(p))
    tested = missing["feature_id"][~np.isnan(p)].tolist()
    named = list(dict.fromkeys(table["symbol"].tolist()))
    some_named = [g for g in dict.fromkeys(unnamed["symbol"].tolist()) if g is not None]
    chosen = [*rng.choice(universe, 200).tolist(), *[f"Y{i}" for i in range(30)]]
    return {
        # kegga's universe of a fit is every gene id of the fit (found by probing).
        "fit": dict(call=(table, sets, {}), lists=selected_lists(table, "feature_id"),
                    universe=ids.tolist(), fit="feature_id"),
        "symbols": dict(call=(table, by_symbol, {"id_column": "symbol"}),
                        lists=selected_lists(table, "symbol"), universe=named,
                        fit="symbol"),
        "missing_symbols": dict(call=(unnamed, by_symbol, {"id_column": "symbol"}),
                                lists=selected_lists(unnamed, "symbol"),
                                universe=some_named, fit="symbol"),
        "restricted_universe": dict(call=(table, sets, {"universe": universe}),
                                    lists=selected_lists(table, "feature_id"),
                                    universe=universe, fit=None),
        "gene_list": dict(call=(chosen, sets, {"universe": universe}),
                          lists={"DE": chosen}, universe=universe, fit=None),
        "missing_p_values": dict(call=(missing, sets, {}),
                                 lists=selected_lists(missing, "feature_id"),
                                 universe=tested, fit="feature_id"),
    }  # fmt: skip


def write_inputs(root, gmt, matching, grid, cases):
    """The files R reads: GMT files as given and as CytePool writes them back, the
    matching inputs, the grid, and every case of kegga."""
    names = []
    for name, data in gmt.items():
        (root / name).write_bytes(data)
        write_gmt(read_gmt(root / name), root / f"written_{name}")
        names += [name, f"written_{name}"]
    (root / "gmt_files.txt").write_text("".join(f"{name}\n" for name in names))
    text, identifiers, y = matching
    (root / "matching.gmt").write_bytes(text)
    (root / "identifiers.txt").write_text("".join(f"{g}\n" for g in identifiers))
    (root / "y.bin").write_bytes(y.tobytes(order="F"))
    (root / "grid.bin").write_bytes(grid.tobytes(order="F"))
    (root / "ora_cases.txt").write_text("".join(f"{name}\n" for name in cases))
    for name, case in cases.items():
        directory = root / name
        directory.mkdir()
        genes, gene_sets, _ = case["call"]
        pairs = [f"{g}\t{s}\n" for s, members in gene_sets.items() for g in members]
        (directory / "gene_pathway.tsv").write_text("".join(pairs))
        lines = {"lists": list(case["lists"]), "universe": case["universe"]}
        lines |= case["lists"]
        if case["fit"] is not None:
            lines["ids"] = ["" if g is None else g for g in genes[case["fit"]].tolist()]
            (directory / "p.bin").write_bytes(np.asarray(genes["p_value"]).tobytes())
            (directory / "lfc.bin").write_bytes(genes["log2_fold_change"].tobytes())
        for file, values in lines.items():
            (directory / f"{file}.txt").write_text("".join(f"{v}\n" for v in values))


def run_references(root):
    """R and decoupler; their versions and decoupler's reading of msigdb.gmt."""
    (root / "compare.R").write_text(R_SCRIPT)
    rscript = os.environ.get("RSCRIPT", "Rscript")
    subprocess.run(
        [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
    )
    program = root / "decoupler_reference.py"
    program.write_text(DECOUPLER_PROGRAM)
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    python = os.environ.get("DECOUPLER_PYTHON", sys.executable)
    command = [
        python,
        "-I",
        str(program),
        str(root / "msigdb.gmt"),
        str(root / "dc.json"),
    ]
    subprocess.run(command, check=True, env=environment)
    decoupler = json.loads((root / "dc.json").read_text())
    versions = (root / "versions.txt").read_text().split("\n")[:2]
    return [*versions, f"decoupler {decoupler['version']}"], decoupler["sets"]


def outputs(directory):
    """What R wrote into a directory, by name: numbers and lines of text."""
    found = {
        path.stem.removeprefix("out "): np.fromfile(path, "<f8")
        for path in directory.glob("out *.bin")
    }
    return found | {
        path.stem.removeprefix("out "): path.read_text().split("\n")[:-1]
        for path in directory.glob("out *.txt")
    }


# Comparisons.


def relative(ours, reference):
    """Largest difference relative to max(|reference|, 1e-300); NaN must match."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    if not np.array_equal(np.isnan(ours), np.isnan(reference)):
        return np.inf
    keep = ~np.isnan(reference)
    scale = np.maximum(np.abs(reference[keep]), 1e-300)
    return float(np.max(np.abs(ours[keep] - reference[keep]) / scale, initial=0.0))


def differing_sets(collection, reading):
    """How many sets differ from a reading in R in name, description or genes."""
    if list(collection) != reading["names"]:
        return np.inf
    return sum(
        collection.descriptions[name] != description or list(collection[name]) != genes
        for name, description, genes in zip(
            reading["names"], reading["descriptions"], reading["genes"], strict=True
        )
    )


def check_files(root, decoupler, table):
    readings = json.loads((root / "out gmt.json").read_text())
    for name in GMT_FILES:
        sets = read_gmt(root / name)
        differ = max(
            differing_sets(sets, readings[name]),
            differing_sets(sets, readings[f"written_{name}"]),
            differing_sets(read_gmt(root / f"written_{name}"), readings[name]),
        )
        check = f"{name}: {len(sets)} sets, read, written and read again"
        table.append((check, "R, by hand", "sets that differ", differ, 0))
    sets = read_gmt(root / "msigdb.gmt")
    reference = {name: list(dict.fromkeys(genes)) for name, genes in decoupler.items()}
    differ = sum(list(sets.get(name, ())) != genes for name, genes in reference.items())
    differ = differ if list(reference) == list(sets) else np.inf
    check = "msigdb.gmt: names and genes of every set"
    table.append((check, "decoupler read_gmt", "sets that differ", differ, 0))


def check_matching(root, matching, reference, table):
    _, identifiers, y = matching
    restricted = read_gmt(root / "matching.gmt").restrict_to(identifiers)
    position = {gene: i + 1 for i, gene in enumerate(identifiers)}
    flat = [sorted(position[g] for g in genes) for genes in restricted.values()]
    same = list(restricted) == reference["index names"] and np.array_equal(
        np.concatenate(flat), reference["index"]
    )
    check = f"restrict_to: {len(restricted)} sets kept, their genes' positions"
    table.append((check, "ids2indices", "differences", int(not same), 0))
    kept = restricted.filter_by_size(15, 200)
    design = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    tested = camera(y.T, kept, design=design, contrast=1, feature_ids=identifiers)
    value = relative(tested["p_value"], reference["camera"])
    value = value if list(kept) == reference["camera names"] else np.inf
    check = f"filter_by_size(15, 200), then camera of the {len(kept)} sets"
    table.append((check, "camera", "max relative difference", value, EXACT))


def check_grid(grid, reference, table):
    ours = hypergeom_sf(grid[:, 0] - 1, grid[:, 3], grid[:, 1], grid[:, 2])
    check = f"P(X >= k) at {len(grid)} points, down to 1e-300 and underflow"
    table.append((check, "phyper", "max relative difference",
                  relative(ours, reference["phyper"]), EXACT))  # fmt: skip


def check_ora(name, case, reference, table):
    """One row per direction and run of kegga; returns CytePool's table."""
    genes, gene_sets, options = case["call"]
    ours = over_representation(genes, gene_sets, **options)
    runs = ["fit ", ""] if case["fit"] is not None else [""]
    for prefix in runs:
        pathways = reference[f"{prefix}pathways"]
        for direction in dict.fromkeys(ours["direction"].tolist()):
            key = LISTS[direction]
            if f"{prefix}P.{key}" not in reference:
                continue
            block = ours.filter(np.array([d == direction for d in ours["direction"]]))
            row = {s: i for i, s in enumerate(block["gene_set"].tolist())}
            at = [row[s] for s in pathways]
            left_out = [row[s] for s in row if s not in set(pathways)]
            # topKEGG's order: by p-value, ties by the number of genes (found by
            # probing).
            order = np.lexsort((block["n_genes"], block["p_value"]))
            top = [block["gene_set"][i] for i in order[: len(pathways)]][:30]
            same = (
                np.array_equal(block["n_genes"][at], reference[f"{prefix}N"])
                and np.array_equal(block["overlap"][at], reference[f"{prefix}{key}"])
                and top == reference[f"{prefix}top {key}"]
                and bool(np.all(np.isnan(block["p_value"][left_out])))
            )
            value = max(
                relative(block["p_value"][at], reference[f"{prefix}P.{key}"]),
                relative(
                    block["adjusted_p_value"][at], reference[f"{prefix}FDR.{key}"]
                ),
            )
            if not prefix:  # kegga's default method: the numbers of genes and expected
                n = reference["selected"][list(case["lists"]).index(key)]
                universe = reference["universe"][0]
                same &= bool(np.all(block["n_selected"] == n))
                same &= bool(np.all(block["n_universe"] == universe))
                expected = reference["N"] * n / universe
                value = max(value, relative(block["expected"][at], expected))
            check = f"{name}: {direction or 'the list'}, {len(pathways)} sets"
            source = "kegga of a fit" if prefix else "kegga"
            if prefix and name in DELIBERATE:
                finite = np.isfinite(block["p_value"][at])  # sets in both tables
                value = relative(
                    block["p_value"][at][finite], reference[f"{prefix}P.{key}"][finite]
                )
                table.append((check + DELIBERATE[name], source,
                              "max relative difference", value, None))  # fmt: skip
            else:
                value = value if same else np.inf
                table.append(
                    (check, f"{source}, p.adjust, topKEGG", "max relative difference",
                     value, EXACT)
                )  # fmt: skip
    return ours


def check_figure(ours, reference, table):
    """The dot plot of the fit case: the sets shown and their order, the overlaps and
    the dots coloured, from kegga's counts and adjusted p-values."""
    ax = plotting.plot_over_representation(
        ours, ax=matplotlib.figure.Figure().subplots()
    )
    dots = next(c for c in ax.collections if c.get_gid() == "gene sets")
    label_at = {
        int(round(y)): label.get_text().rsplit(" (", 1)[0]
        for y, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)
    }
    runs = (("fit ", "Up"), ("fit ", "Down"), ("", "Either"))  # the columns, in order
    best = {}
    for prefix, key in runs:
        for s, p in zip(reference[f"{prefix}pathways"], reference[f"{prefix}P.{key}"]):
            best[s] = min(best.get(s, np.inf), p)
    shown = [label_at[y] for y in sorted(label_at, reverse=True)]
    counts, coloured = [], []
    for x, y in dots.get_offsets():
        prefix, key = runs[int(round(x))]
        i = reference[f"{prefix}pathways"].index(label_at[int(round(y))])
        counts.append(reference[f"{prefix}{key}"][i])
        coloured.append(reference[f"{prefix}FDR.{key}"][i] < ALPHA)
    sizes = dots.get_sizes()
    pale = np.all(dots.get_facecolors() == matplotlib.colors.to_rgba("#e1e0d9"), axis=1)
    same = (
        shown == sorted(best, key=best.__getitem__)[:20]
        and np.allclose(sizes / sizes.max(), np.divide(counts, max(counts)), rtol=1e-12)
        and np.array_equal(~pale, coloured)
    )
    check = f"plot_over_representation: {len(shown)} sets, {len(counts)} dots"
    table.append((check, "kegga", "differences", int(not same), 0))


def check_left_out(cases, references, table):
    """The deliberate difference in the table: sets without genes in the universe."""
    left_out = {
        name: len(case["call"][1]) - len(references[name]["pathways"])
        for name, case in cases.items()
    }
    check = "sets without genes in the universe: CytePool's rows with NaN p-values"
    value = ", ".join(f"{n} ({name})" for name, n in left_out.items())
    table.append((check, "kegga", "sets kegga leaves out", value, None))


# Fixtures.


def write_fixtures(root, gmt, matching, grid, cases, references, top):
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    for name in gmt:
        shutil.copy(root / name, FIXTURES / name)
    shutil.copy(root / "out gmt.json", FIXTURES / "gmt_by_hand.json")
    text, identifiers, y = matching
    (FIXTURES / "matching.gmt").write_bytes(text)
    found = ("index names", "index", "index sizes", "camera names", "camera")
    np.savez_compressed(
        FIXTURES / "matching.npz", identifiers=np.array(identifiers), y=y,
        **{f"limma {key}": np.asarray(top[key]) for key in found},
    )  # fmt: skip
    np.savez_compressed(FIXTURES / "phyper.npz", grid=grid, phyper=top["phyper"])
    for name, case in cases.items():
        genes, gene_sets, options = case["call"]
        arrays = {
            "set_names": np.array(list(gene_sets)),
            "set_sizes": np.array([len(g) for g in gene_sets.values()]),
            "set_genes": np.array(
                [g for members in gene_sets.values() for g in members]
            ),
        }
        if isinstance(genes, Table):
            arrays |= {f"table {column}": np.array(
                ["" if v is None else v for v in genes[column].tolist()]
                if genes[column].dtype == object else genes[column]
            ) for column in genes.columns}  # fmt: skip
        else:
            arrays["genes"] = np.array(genes)
        for key, value in options.items():
            arrays[f"option {key}"] = np.array(value)
        arrays |= {f"kegga {k}": np.array(v) for k, v in references[name].items()}
        np.savez_compressed(FIXTURES / f"ora_{name}.npz", **arrays)
    print(f"wrote the fixtures to {FIXTURES}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    small = arguments.write_fixtures
    rng = np.random.default_rng(20261026)
    gmt = gmt_files(rng, 6 if small else 60)
    matching = matching_inputs(rng, 300 if small else 2000)
    grid = hypergeometric_grid()
    cases = ora_cases(rng, 400 if small else 4000, 40 if small else 300)
    table = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_inputs(root, gmt, matching, grid, cases)
        versions, decoupler = run_references(root)
        print("Reference: " + ", ".join(versions))
        top = outputs(root)
        references = {name: outputs(root / name) for name in cases}
        check_files(root, decoupler, table)
        check_matching(root, matching, top, table)
        check_grid(grid, top, table)
        tables = {
            name: check_ora(name, case, references[name], table)
            for name, case in cases.items()
        }
        check_figure(tables["fit"], references["fit"], table)
        check_left_out(cases, references, table)
        failures = 0
        print("| check | reference | metric | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- | --- |")
        for check, source, metric, value, tolerance in table:
            if tolerance is None:
                result, shown = "deliberate difference", "n/a"
            else:
                result = "pass" if value <= tolerance else "FAIL"
                shown = f"{tolerance:.0e}" if tolerance else "0"
            failures += result == "FAIL"
            if isinstance(value, float):
                value = f"{value:.1e}"
            print(f"| {check} | {source} | {metric} | {value} | {shown} | {result} |")
        print(f"{len(table)} checks, {failures} failed")
        if small:
            if failures:
                print("not writing fixtures: some checks failed")
            else:
                write_fixtures(root, gmt, matching, grid, cases, references, top)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
