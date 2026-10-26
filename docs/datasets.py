"""The datasets of the documentation, prepared once and kept outside the repository.

Several pages read the same data: a simulated single-cell study (its cells, and the
cube pooled from them), a simulated bulk RNA-seq experiment, the published
interferon-beta stimulation of Kang et al. (2018), and the MSigDB Hallmark gene set
of the interferon-alpha response. This script makes them deterministically into a
cache outside the repository and records, for each, where it comes from, its
licence, how to cite it, the SHA-256 of every file it downloads, and what it must
contain: the fingerprint of every cube it writes and the digest of every other file.
Pages open the cubes with ``read_cube``.

Usage, from the repository root::

    uv run --group docs python docs/datasets.py list
    uv run --group docs python docs/datasets.py prepare [ids]
    uv run --group docs python docs/datasets.py verify [ids]

``prepare`` makes the datasets missing from the cache, downloading what they need,
and checks each against the registry; ``verify`` makes nothing and recomputes every
fingerprint and digest from the files, so it also notices a cache that changed or a
generator whose random streams changed (numpy does not promise the same streams in
every release). Without ids, every dataset.

The cache is ``CYTEPOOL_DOCS_DATA``, else ``cytepool-docs`` under ``XDG_CACHE_HOME``
or ``~/.cache``. The GEO files of Kang et al. (2018) are those that
``validation/interferon_stimulated_pbmcs.py`` downloads, with the same checksums and
in the same directory (``GSE96583_DATA``, else ``cytepool-validation/GSE96583`` under
the same cache root), so a machine downloads them once.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cytepool.cube import from_bulk, from_cells
from cytepool.io import read_cube, write_cube

REPOSITORY = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Download:
    """A file to download, with its size in bytes and its SHA-256."""

    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Dataset:
    """A dataset: where it comes from, its terms, and what preparing it writes.

    ``files`` gives each file written into the cache with its expected content: the
    fingerprint of a cube file (``Cube.fingerprint()``), or the ``sha256:`` digest of
    the arrays of an ``.npz`` file or of the text of a ``.json`` file (``digest``).
    """

    id: str
    title: str
    description: str
    source: str
    licence: str
    citation: str | None  # a key of docs/source/references.bib
    files: dict[str, str]
    build: Callable[[Path, Path], None] = field(repr=False)
    downloads: dict[str, Download] = field(default_factory=dict)


# Simulations.

SIMULATION_LICENCE = (
    "Simulated by this script, which is part of CytePool and distributed under the "
    "BSD 3-Clause license"
)
CELL_TYPES = (
    "B cells",
    "CD4 T cells",
    "CD8 T cells",
    "NK cells",
    "Monocytes",
    "Dendritic cells",
)
# The share of a donor's cells in each type before stimulation; dendritic cells are
# rare enough that some samples have fewer than ten.
CELL_TYPE_SHARES = (0.12, 0.34, 0.15, 0.10, 0.27, 0.02)
# Stimulation multiplies the odds of monocytes, against the other types, by e^0.7.
MONOCYTE_LOG_ODDS_CHANGE = 0.7


def simulated_pbmc(downloads: Path, output: Path) -> None:
    """A study of 12 donors, each sampled before and after a stimulation.

    About 20,000 cells of six cell types over 2,000 genes. Each cell type has its
    own expression profile with marker genes; each donor shifts every gene a little
    in all its cells; the stimulation changes 100 genes of each cell type (a
    different set for each), raises the 40 genes of a response program in every
    cell type, and raises the share of monocytes. Cells vary in depth (log-normal
    around 3,000 counts) and around their type's profile (gamma-Poisson counts,
    dispersion 0.2). Writes the cube of all five layers and the cells, with the
    true effects.
    """
    rng = np.random.default_rng(20261026)
    n_genes, n_donors = 2_000, 12
    genes = np.array([f"gene_{g:04d}" for g in range(1, n_genes + 1)])
    donors = [f"D{d:02d}" for d in range(1, n_donors + 1)]
    sexes = ["female", "male"] * (n_donors // 2)
    n_types = len(CELL_TYPES)

    base = rng.normal(0.0, 1.6, n_genes)
    profiles = base + rng.normal(0.0, 0.6, (n_types, n_genes))
    markers = np.zeros((n_types, n_genes), dtype=bool)
    for t in range(n_types):
        chosen = rng.choice(np.flatnonzero(~markers.any(axis=0)), 20, replace=False)
        markers[t, chosen] = True
        profiles[t, chosen] += rng.uniform(2.0, 4.0, 20)
    response = rng.choice(np.flatnonzero(~markers.any(axis=0)), 40, replace=False)
    log2_fc = np.zeros((n_types, n_genes))
    for t in range(n_types):
        free = np.setdiff1d(np.arange(n_genes), response)
        changed = rng.choice(free, 100, replace=False)
        log2_fc[t, changed] = rng.choice([-1.0, 1.0], 100) * rng.uniform(0.7, 2.0, 100)
        log2_fc[t, response] = rng.uniform(1.0, 2.5, response.size)
    donor_shift = rng.normal(0.0, 0.25, (n_donors, n_genes))
    donor_composition = rng.normal(0.0, 0.25, (n_donors, n_types))
    composition_change = np.zeros(n_types)
    composition_change[CELL_TYPES.index("Monocytes")] = MONOCYTE_LOG_ODDS_CHANGE

    blocks, cell_samples, cell_types = [], [], []
    for d, donor in enumerate(donors):
        for condition in ("ctrl", "stim"):
            stim = condition == "stim"
            log_odds = (
                np.log(CELL_TYPE_SHARES)
                + donor_composition[d]
                + stim * composition_change
            )
            shares = np.exp(log_odds) / np.exp(log_odds).sum()
            counts_per_type = rng.multinomial(rng.poisson(800), shares)
            for t, n_cells in enumerate(counts_per_type):
                log2_mean = profiles[t] + donor_shift[d] + stim * log2_fc[t]
                proportion = 2.0**log2_mean / np.sum(2.0**log2_mean)
                depth = rng.lognormal(np.log(3_000.0), 0.4, (n_cells, 1))
                rate = rng.gamma(5.0, 0.2, (n_cells, n_genes)) * depth * proportion
                blocks.append(_csr(rng.poisson(rate)))
                cell_samples += [f"{donor}_{condition}"] * n_cells
                cell_types += [CELL_TYPES[t]] * n_cells
    cells = _stack(blocks, n_genes)
    sample_table = {
        "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
        "donor": [d for d in donors for _ in range(2)],
        "condition": ["ctrl", "stim"] * n_donors,
        "sex": [s for s in sexes for _ in range(2)],
    }
    features = {"feature_id": genes, "symbol": genes}
    cube = from_cells(
        SimpleNamespace(**cells),
        samples=cell_samples,
        groups=cell_types,
        sample_table=sample_table,
        group_table={"group_id": list(CELL_TYPES)},
        features=features,
        source={
            "dataset": "simulated_pbmc",
            "generator": "docs/datasets.py, numpy.random.default_rng(20261026)",
            "cells": "every simulated cell",
            "samples": "donor_condition",
            "groups": "cell type",
        },
    )
    write_cube(cube, output / "simulated_pbmc.cube.zarr", overwrite=True)
    np.savez_compressed(
        output / "simulated_pbmc_cells.npz",
        data=cells["data"],
        indices=cells["indices"],
        indptr=cells["indptr"],
        shape=np.array(cells["shape"]),
        cell_sample=np.array(cell_samples),
        cell_type=np.array(cell_types),
        feature_id=genes,
        cell_types=np.array(CELL_TYPES),
        true_log2_fold_change=log2_fc,
        response_genes=genes[np.sort(response)],
        marker_genes=np.array([genes[markers[t]] for t in range(n_types)]),
        true_composition_change=composition_change,
    )


def simulated_bulk(downloads: Path, output: Path) -> None:
    """A bulk RNA-seq experiment: three treatments, four replicates in two batches.

    10,000 genes with log-normal abundances and negative binomial counts whose
    dispersions scatter around 0.05; library sizes from 15 to 25 million reads.
    Drug A changes 500 genes and drug B 300, 150 of them shared, by 0.5 to 2 on the
    log2 scale; the batches shift every gene a little. One sample, drug_b_3, is
    noisier than the others, with four times the dispersion, as a degraded sample
    would be. Writes the cube and the true effects.
    """
    rng = np.random.default_rng(20261027)
    n_genes = 10_000
    genes = np.array([f"gene_{g:05d}" for g in range(1, n_genes + 1)])
    conditions = ["control", "drug_a", "drug_b"]
    samples = [(c, r) for c in conditions for r in range(1, 5)]
    abundance = rng.lognormal(0.0, 2.0, n_genes)
    abundance /= abundance.sum()
    dispersion = 0.05 * rng.lognormal(0.0, 0.5, n_genes)
    drug_a = rng.choice(n_genes, 500, replace=False)
    shared = rng.choice(drug_a, 150, replace=False)
    only_b = rng.choice(np.setdiff1d(np.arange(n_genes), drug_a), 150, replace=False)
    log2_fc = {"drug_a": np.zeros(n_genes), "drug_b": np.zeros(n_genes)}
    log2_fc["drug_a"][drug_a] = rng.choice([-1.0, 1.0], 500) * rng.uniform(
        0.5, 2.0, 500
    )
    drug_b = np.concatenate([shared, only_b])
    log2_fc["drug_b"][drug_b] = rng.choice([-1.0, 1.0], 300) * rng.uniform(
        0.5, 2.0, 300
    )
    batch_shift = rng.normal(0.0, 0.15, (2, n_genes))
    outlier = "drug_b_3"
    counts = np.empty((len(samples), n_genes), dtype=np.int64)
    for i, (condition, replicate) in enumerate(samples):
        batch = (replicate - 1) // 2
        log2_effect = log2_fc.get(condition, np.zeros(n_genes)) + batch_shift[batch]
        mean = rng.uniform(15e6, 25e6) * abundance * 2.0**log2_effect
        phi = dispersion * (4.0 if f"{condition}_{replicate}" == outlier else 1.0)
        counts[i] = rng.poisson(rng.gamma(1.0 / phi, mean * phi))
    sample_table = {
        "sample_id": [f"{c}_{r}" for c, r in samples],
        "condition": [c for c, _ in samples],
        "batch": [f"batch{(r - 1) // 2 + 1}" for _, r in samples],
    }
    cube = from_bulk(
        counts,
        samples=sample_table,
        features={"feature_id": genes, "symbol": genes},
        source={
            "dataset": "simulated_bulk",
            "generator": "docs/datasets.py, numpy.random.default_rng(20261027)",
        },
    )
    write_cube(cube, output / "simulated_bulk.cube.zarr", overwrite=True)
    np.savez_compressed(
        output / "simulated_bulk_truth.npz",
        feature_id=genes,
        true_log2_fold_change_drug_a=log2_fc["drug_a"],
        true_log2_fold_change_drug_b=log2_fc["drug_b"],
        outlying_sample=np.array(outlier),
    )


# Kang et al. (2018): GEO series GSE96583, batch 2.

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/"
GEO_FILES = {  # the files and checksums of validation/interferon_stimulated_pbmcs.py
    "GSE96583_RAW.tar": (
        76_195_840,
        "e5d41a3248a813f99d68fd5c9eb9773de7f46a83680a67f4a02d683b8955fe80",
    ),
    "GSE96583_batch2.genes.tsv.gz": (
        277_054,
        "93aa4e9b530ef9d6411ca129b416324c5cc1cc5a01a1fa6ed4f4a845480ed3ca",
    ),
    "GSE96583_batch2.total.tsne.df.tsv.gz": (
        756_342,
        "1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09",
    ),
}
# Per condition: the counts (genes x cells) in the archive, its barcodes, and the
# numbers of cells and of stored counts.
GEO_MATRICES = {
    "ctrl": ("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz", 14_619, 8_732_747),
    "stim": ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz", 14_446, 8_838_098),
}
GEO_GENES = 35_635
GEO_CELL_TYPES = {
    "CD4 T cells": 10_389,
    "CD14+ Monocytes": 5_385,
    "B cells": 2_573,
    "CD8 T cells": 2_042,
    "NK cells": 1_993,
    "FCGR3A+ Monocytes": 1_599,
    "Dendritic cells": 432,
    "Megakaryocytes": 260,
}


def kang2018(downloads: Path, output: Path) -> None:
    """Interferon-beta stimulated PBMCs of 8 lupus patients (Kang et al. 2018).

    Each donor's cells were split into a culture stimulated with interferon-beta
    for 6 hours and an untreated one; the cells of all donors were pooled and
    sequenced together, and assigned to donors by their genotypes. The cube pools
    the 24,673 singlets that have a cell type in the authors' cell table into 16
    samples (8 donors x 2 conditions) x 8 cell types x 35,635 genes, keyed by
    Ensembl id with symbols, with the counts and the number of detecting cells.
    Every cell is matched to the table by condition and barcode, never by position.
    """
    with gzip.open(downloads / "GSE96583_batch2.genes.tsv.gz", "rt") as text:
        genes = [line.rstrip("\n").split("\t") for line in text]
    _require(
        len(genes) == GEO_GENES and {len(gene) for gene in genes} == {2},
        f"the genes file lists {GEO_GENES:,} genes, each an Ensembl id and a symbol",
    )
    with gzip.open(downloads / "GSE96583_batch2.total.tsne.df.tsv.gz", "rt") as text:
        header, *lines = text.read().splitlines()
    table = {name: rest for name, *rest in (line.split("\t") for line in lines)}
    _require(
        header.split("\t")[2:] == ["ind", "stim", "cluster", "cell", "multiplets"]
        and len(table) == len(lines),
        "the cell table has one row per barcode with ind, stim, cell and multiplets",
    )
    blocks, matched = [], Counter()
    labels: dict[str, list[str]] = {"sample": [], "group": []}
    with tarfile.open(downloads / "GSE96583_RAW.tar") as archive:
        barcodes = {}
        for condition, (_, member, n_cells, _) in GEO_MATRICES.items():
            with archive.extractfile(member) as raw:  # type: ignore[union-attr]
                barcodes[condition] = gzip.decompress(raw.read()).decode().split()
            _require(
                len(set(barcodes[condition])) == n_cells,
                f"{member} has {n_cells:,} barcodes",
            )
        shared = set(barcodes["ctrl"]) & set(barcodes["stim"])
        for condition, (member, _, n_cells, stored) in GEO_MATRICES.items():
            with archive.extractfile(member) as raw:  # type: ignore[union-attr]
                shape, entries = _matrix_market(raw)
            _require(
                shape == (GEO_GENES, n_cells) and len(entries) == stored,
                f"{member} holds {GEO_GENES:,} genes x {n_cells:,} cells",
            )
            # The cell table adds a 1 to a stimulated barcode that is also a control
            # one.
            names = [
                f"{barcode}1" if condition == "stim" and barcode in shared else barcode
                for barcode in barcodes[condition]
            ]
            matched.update(names)
            rows = [table.get(name) for name in names]
            _require(
                all(row is not None and row[3] == condition for row in rows),
                f"every {condition} barcode names a {condition} cell of the table",
            )
            keep = np.array(
                [row[6] == "singlet" and row[5] != "NA" for row in rows]  # type: ignore[index]
            )
            cells = _cells_from_entries(entries, n_cells, GEO_GENES)
            blocks.append(_take_rows(cells, np.flatnonzero(keep)))
            for position in np.flatnonzero(keep):
                row = rows[position]
                labels["sample"].append(f"{row[2]}_{condition}")  # type: ignore[index]
                labels["group"].append(row[5])  # type: ignore[index]
    _require(
        set(matched) == set(table) and set(matched.values()) == {1},
        "every cell of the table matches one barcode, and every barcode one cell",
    )
    _require(
        Counter(labels["group"]) == GEO_CELL_TYPES, f"the singlets are {GEO_CELL_TYPES}"
    )
    donors = sorted({sample.split("_")[0] for sample in labels["sample"]})
    cube = from_cells(
        SimpleNamespace(**_stack(blocks, GEO_GENES)),
        samples=labels["sample"],
        groups=labels["group"],
        sample_table={
            "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
            "donor": [d for d in donors for _ in range(2)],
            "condition": ["ctrl", "stim"] * len(donors),
        },
        features={"feature_id": [g[0] for g in genes], "symbol": [g[1] for g in genes]},
        layers=["counts", "detected"],
        source={
            "dataset": "GEO series GSE96583, batch 2 (Kang et al. 2018)",
            "doi": "10.1038/nbt.4042",
            "files": {
                name: f"sha256:{checksum}" for name, (_, checksum) in GEO_FILES.items()
            },
            "cells": "the singlets of GSE96583_batch2.total.tsne.df.tsv.gz that have a "
            "cell type (multiplets == 'singlet' and cell != 'NA')",
            "samples": "the table's ind (donor) and stim (condition), as donor_condition",
            "groups": "the table's cell column",
            "prepared_by": "docs/datasets.py",
        },
    )
    write_cube(cube, output / "kang2018.cube.zarr", overwrite=True)


# MSigDB's Hallmark gene set HALLMARK_INTERFERON_ALPHA_RESPONSE (M5911, v2026.2.Hs), as
# validation/interferon_stimulated_pbmcs.py lists it; two members have earlier symbols
# in the annotation of GSE96583.
INTERFERON_ALPHA_RESPONSE = (
    "ADAR B2M BATF2 BST2 C1S CASP1 CASP8 CCRL2 CD47 CD74 CMPK2 CMTR1 CNP CSF1 CXCL10 "
    "CXCL11 DDX60 DHX58 EIF2AK2 ELF1 EPSTI1 GBP2 GBP4 GMPR HELZ2 HERC6 HLA-C IFI27 "
    "IFI30 IFI35 IFI44 IFI44L IFIH1 IFIT2 IFIT3 IFITM1 IFITM2 IFITM3 IL15 IL4R IL7 "
    "IRF1 IRF2 IRF7 IRF9 ISG15 ISG20 LAMP3 LAP3 LGALS3BP LPAR6 LY6E MOV10 MVB12A MX1 "
    "NCOA7 NMI NUB1 OAS1 OASL OGFR PARP12 PARP14 PARP9 PLSCR1 PNPT1 PROCR PSMA3 PSMB8 "
    "PSMB9 PSME1 PSME2 RIPK2 RNF31 RSAD2 RTP4 SAMD9 SAMD9L SELL SLC25A28 SP110 STAT2 "
    "TAP1 TDRD7 TENT5A TMEM140 TRAFD1 TRIM14 TRIM21 TRIM25 TRIM26 TRIM5 TXNIP UBA7 "
    "UBE2L6 USP18 WARS1"
).split()
EARLIER_SYMBOLS = {"TENT5A": "FAM46A", "WARS1": "WARS"}


def hallmark_interferon_alpha(downloads: Path, output: Path) -> None:
    """The 97 genes of MSigDB's Hallmark interferon-alpha response set."""
    record = {
        "name": "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "systematic_name": "M5911",
        "release": "MSigDB v2026.2.Hs",
        "source": "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/"
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "genes": INTERFERON_ALPHA_RESPONSE,
        "earlier_symbols": EARLIER_SYMBOLS,
    }
    path = output / "hallmark_interferon_alpha_response.json"
    path.write_text(json.dumps(record, indent=1) + "\n")


GEO_TERMS = (
    "NCBI places no restrictions on the use or distribution of the GEO data; some "
    "submitters may claim patent, copyright or other rights in the data they "
    "submitted (https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html)"
)
MSIGDB_TERMS = (
    "Creative Commons Attribution 4.0 International (MSigDB v2022.1 and above); the "
    "contents of the set are protected by copyright (c) 2004-2026 Broad Institute, "
    "Inc., Massachusetts Institute of Technology, and Regents of the University of "
    "California (https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp)"
)

DATASETS = {
    dataset.id: dataset
    for dataset in (
        Dataset(
            id="simulated_pbmc",
            title="Simulated PBMC study",
            description="12 donors x 2 conditions x 6 cell types, about 20,000 cells "
            "and 2,000 genes, with planted effects",
            source="simulated by simulated_pbmc() in this script, numpy's PCG64 "
            "generator with seed 20261026",
            licence=SIMULATION_LICENCE,
            citation=None,
            files={
                "simulated_pbmc.cube.zarr": "sha256:"
                "554a2b5b4188308ea900cf907fa87f42d12e0458e6d5dc11525e862f3ad8a726",
                "simulated_pbmc_cells.npz": "sha256:"
                "0d4c9154b87dbf8e9796232c2afec4a8340368fd0f5d386df4dd780c4e478129",
            },
            build=simulated_pbmc,
        ),
        Dataset(
            id="simulated_bulk",
            title="Simulated bulk RNA-seq experiment",
            description="3 treatments x 4 replicates in 2 batches, 10,000 genes, "
            "one outlying sample",
            source="simulated by simulated_bulk() in this script, numpy's PCG64 "
            "generator with seed 20261027",
            licence=SIMULATION_LICENCE,
            citation=None,
            files={
                "simulated_bulk.cube.zarr": "sha256:"
                "ac41de9421f1171ef9273f8d3104a6ff2e5b7fe34a503d29ad63e11da4770cd2",
                "simulated_bulk_truth.npz": "sha256:"
                "a8a0275532e3cafb02401846176a7ddfa037579aceb804c5f9c007700f022977",
            },
            build=simulated_bulk,
        ),
        Dataset(
            id="kang2018",
            title="Interferon-beta stimulated PBMCs (Kang et al. 2018)",
            description="16 samples (8 donors x 2 conditions) x 8 cell types x 35,635 "
            "genes, from 24,673 cells; a download of 77 MB",
            source="GEO series GSE96583 (batch 2), "
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583",
            licence=GEO_TERMS,
            citation="kang2018",
            files={
                "kang2018.cube.zarr": "sha256:"
                "a30af3dabe17f21a6c75f41b64a279a1423640b9aa919022cf67897f9cc77a51",
            },
            build=kang2018,
            downloads={
                name: Download(GEO + name, size, checksum)
                for name, (size, checksum) in GEO_FILES.items()
            },
        ),
        Dataset(
            id="hallmark_interferon_alpha",
            title="Hallmark interferon-alpha response genes (MSigDB)",
            description="the 97 gene symbols of HALLMARK_INTERFERON_ALPHA_RESPONSE",
            source="MSigDB v2026.2.Hs, https://www.gsea-msigdb.org/gsea/msigdb/human/"
            "geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE",
            licence=MSIGDB_TERMS,
            citation="liberzon2015",
            files={
                "hallmark_interferon_alpha_response.json": "sha256:"
                "c5ef439a3f9b6e7faef95eb6b9f7c82ed1a48ede2af78629c85be3b873a8f1c8",
            },
            build=hallmark_interferon_alpha,
        ),
    )
}


# The cache.


def cache_root() -> Path:
    """The user cache directory: ``XDG_CACHE_HOME``, else ``~/.cache``."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def data_directory() -> Path:
    """The directory the datasets are written to, outside the repository."""
    configured = os.environ.get("CYTEPOOL_DOCS_DATA")
    directory = (
        Path(configured).expanduser() if configured else cache_root() / "cytepool-docs"
    )
    return _outside_repository(directory)


def download_directory(dataset: Dataset) -> Path:
    """Where the downloads of a dataset are kept: for GEO's files, the directory of
    the validation script, so that a machine downloads them once."""
    if dataset.id == "kang2018":
        configured = os.environ.get("GSE96583_DATA")
        directory = (
            Path(configured).expanduser()
            if configured
            else cache_root() / "cytepool-validation" / "GSE96583"
        )
    else:
        directory = data_directory() / "downloads" / dataset.id
    return _outside_repository(directory)


def _outside_repository(directory: Path) -> Path:
    if directory.resolve().is_relative_to(REPOSITORY):
        raise SystemExit(
            f"{directory} is inside the repository; keep the data outside it"
        )
    return directory


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(dataset: Dataset) -> Path:
    """Download the files a dataset needs that are missing, keeping a download only
    when its SHA-256 is right, and check those already there."""
    directory = download_directory(dataset)
    if not dataset.downloads:
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    for name, download in dataset.downloads.items():
        path = directory / name
        if path.exists():
            found = sha256(path)
            _require(
                found == download.sha256,
                f"{path} has SHA-256 {found}, not {download.sha256}; delete it to "
                "download it again",
            )
            continue
        print(f"downloading {download.url} ({download.size:,} bytes)", flush=True)
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(download.url, timeout=120) as response:
            with partial.open("wb") as file:
                shutil.copyfileobj(response, file, 1 << 20)
        found = sha256(partial)
        if found != download.sha256:
            partial.unlink()
        _require(
            found == download.sha256, f"the download of {name} has SHA-256 {found}"
        )
        partial.replace(path)
    return directory


def digest(path: Path) -> str:
    """The content digest of a file the registry records.

    A cube file's fingerprint, recomputed from its values
    (``Cube.verify_fingerprint()``). An ``.npz`` file's arrays, in the order of their
    names, each as its name, dtype, shape and bytes, which do not depend on the
    archive's timestamps. Other files' bytes.
    """
    if path.name.endswith(".cube.zarr"):
        return read_cube(path).verify_fingerprint()
    hasher = hashlib.sha256()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            for name in sorted(arrays.files):
                array = np.ascontiguousarray(arrays[name])
                header = json.dumps([name, array.dtype.str, list(array.shape)])
                hasher.update(len(header).to_bytes(8, "little") + header.encode())
                hasher.update(array.tobytes())
    else:
        hasher.update(path.read_bytes())
    return "sha256:" + hasher.hexdigest()


def check(dataset: Dataset, directory: Path) -> list[str]:
    """The files of a dataset that are missing or differ from the registry."""
    problems = []
    for name, expected in dataset.files.items():
        path = directory / name
        if not path.exists():
            problems.append(
                f"{name} is missing; run: python docs/datasets.py prepare {dataset.id}"
            )
            continue
        found = digest(path)
        if found != expected:
            problems.append(f"{name} has {found}, the registry expects {expected}")
    return problems


def prepare(dataset: Dataset, directory: Path) -> list[str]:
    """Make a dataset unless the cache holds it already, and check it."""
    if all((directory / name).exists() for name in dataset.files):
        return check(dataset, directory)
    downloads = fetch(dataset)
    directory.mkdir(parents=True, exist_ok=True)
    dataset.build(downloads, directory)
    return check(dataset, directory)


def path(dataset_id: str, file: str | None = None) -> Path:
    """The path of a prepared file of a dataset (its first file by default)."""
    dataset = DATASETS[dataset_id]
    return data_directory() / (file or next(iter(dataset.files)))


# Reading counts without scipy.


def _require(condition: bool, fact: str) -> None:
    if not condition:
        raise SystemExit(f"The data do not match what this script expects: {fact}")


def _matrix_market(raw: Any) -> tuple[tuple[int, int], np.ndarray]:
    """The shape and the (row, column, value) entries, counted from 1, of a gzipped
    Matrix Market file in coordinate format whose values are whole numbers (GEO's
    files declare them ``real``)."""
    with gzip.open(raw, "rt") as stream:
        banner = stream.readline().split()
        _require(
            banner[:3] == ["%%MatrixMarket", "matrix", "coordinate"]
            and banner[3:] in (["integer", "general"], ["real", "general"]),
            "the counts are in Matrix Market coordinate format",
        )
        line = stream.readline()
        while line.startswith("%"):
            line = stream.readline()
        n_rows, n_columns, _ = map(int, line.split())
        entries = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    _require(
        bool(np.all(entries >= 0) and np.all(entries == np.floor(entries))),
        "the counts and their positions are whole numbers",
    )
    return (n_rows, n_columns), entries.astype(np.int64)


def _cells_from_entries(
    entries: np.ndarray, n_cells: int, n_genes: int
) -> dict[str, Any]:
    """Cells x genes CSR arrays from genes x cells coordinates counted from 1."""
    genes, cells, values = entries[:, 0] - 1, entries[:, 1] - 1, entries[:, 2]
    order = np.lexsort((genes, cells))
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(cells, minlength=n_cells), out=indptr[1:])
    return {
        "data": values[order],
        "indices": genes[order],
        "indptr": indptr,
        "shape": (n_cells, n_genes),
    }


def _take_rows(cells: dict[str, Any], rows: np.ndarray) -> dict[str, Any]:
    starts, ends = cells["indptr"][rows], cells["indptr"][rows + 1]
    positions = np.concatenate(
        [np.arange(start, end) for start, end in zip(starts, ends, strict=True)]
    )
    indptr = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(ends - starts, out=indptr[1:])
    return {
        "data": cells["data"][positions],
        "indices": cells["indices"][positions],
        "indptr": indptr,
        "shape": (rows.size, cells["shape"][1]),
    }


def _csr(dense: np.ndarray) -> dict[str, Any]:
    rows, columns = np.nonzero(dense)
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=dense.shape[0]), out=indptr[1:])
    return {
        "data": dense[rows, columns].astype(np.int64),
        "indices": columns.astype(np.int64),
        "indptr": indptr,
        "shape": dense.shape,
    }


def _stack(blocks: list[dict[str, Any]], n_columns: int) -> dict[str, Any]:
    """CSR arrays of the rows of several CSR blocks, one after another."""
    offsets = np.cumsum([0] + [block["indptr"][-1] for block in blocks])
    indptr = np.concatenate(
        [[0]]
        + [
            block["indptr"][1:] + offset
            for block, offset in zip(blocks, offsets, strict=False)
        ]
    )
    return {
        "data": np.concatenate([block["data"] for block in blocks]),
        "indices": np.concatenate([block["indices"] for block in blocks]),
        "indptr": indptr.astype(np.int64),
        "shape": (int(sum(block["shape"][0] for block in blocks)), n_columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["list", "prepare", "verify"])
    parser.add_argument("ids", nargs="*", help="datasets (default: all)")
    arguments = parser.parse_args()
    unknown = sorted(set(arguments.ids) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; choose from {sorted(DATASETS)}")
    chosen = [DATASETS[i] for i in arguments.ids or DATASETS]
    directory = data_directory()
    if arguments.command == "list":
        for dataset in chosen:
            ready = all((directory / name).exists() for name in dataset.files)
            print(
                f"{dataset.id}: {dataset.title} ({'prepared' if ready else 'not prepared'})"
            )
            print(f"  {dataset.description}")
            print(f"  source: {dataset.source}")
            print(f"  licence: {dataset.licence}")
            if dataset.citation:
                print(f"  cite: [@{dataset.citation}] in docs/source/references.bib")
            for name, download in dataset.downloads.items():
                print(
                    f"  download: {name}, {download.size:,} bytes, sha256 {download.sha256}"
                )
            for name, expected in dataset.files.items():
                print(f"  file: {directory / name} {expected}")
        return
    failures = []
    for dataset in chosen:
        if arguments.command == "prepare":
            problems = prepare(dataset, directory)
        else:
            problems = check(dataset, directory)
        print(f"{dataset.id}: {'; '.join(problems) if problems else 'as registered'}")
        failures += problems
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
