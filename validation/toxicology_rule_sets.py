"""Check NTP's and ETAP's rule sets of dose_response, Williams' trend test and BMDS's
recommended model against BMDExpress 2.30.520, pybmds 25.2, scipy and R.

Data: the liver samples of three chemicals of GEO series GSE147072 (Gwinn et al. 2020,
Toxicological Sciences 176, 343-354), TempO-Seq counts of the rat S1500+ probes of male
Sprague Dawley rats dosed by gavage for 5 days, which the study names a potent
hepatotoxicant (DE-71: the vehicle x 3 and nine doses x 4), a toxicant with minimal
hepatic effects (acrylamide) and an extract with little overt toxicity (milk thistle
extract), each the vehicle and eight doses x 4. The script downloads the counts and the
series matrix (5.7 MB and 61 KB) into a directory outside the repository and checks
their SHA-256; the doses are GEO's (mg/kg). Each chemical is a cube of one group,
"liver", analysed with ``Design(samples, "~ 1")``.

References, each a separate program that sees the counts or values, the doses and the
settings, and nothing CytePool computed:

- The reference process computes the values by ETAP's text (probes with a mean count of
  at least 5, counts per million over them, log2(CPM + 1)) and writes them to BMDExpress
  as the float32 values BMDExpress stores. BMDExpress 2.30.520 (``BMDEXPRESS``, its
  ``bmdexpress2-cmd``) runs ``analyze`` twice: its ANOVA and Williams prefilters of every
  probe (10,000 permutations) and the rule sets' prefilter; then its BMD analyses under
  each rule set's settings, with ETAP's and NTP's gene filters on one category per
  probe, of the first 100 probes its prefilter keeps for each chemical (fewer where it
  keeps fewer) and of 33 genes simulated in the script to push each model's
  parameters towards their bounds. scipy 1.18.1 computes the one-way ANOVA
  (``f_oneway``) and isotonic regression (``isotonic_regression``).
- Williams' trend test on seeded small designs (4 x 2, 3, 2, 2, 2 and 3 x 3; 60 genes
  each): the exact p-value over every distinct relabelling, by the minimax formula of
  isotonic regression.
- pybmds 25.2 (``BMDS_PYTHON``) fits BMDS's six continuous models to the subset's probes
  and 60 seeded genes, under constant variance at 1 SD, and recommends a model.
- R 4.4 (``RSCRIPT``) codes both documents' acceptance from their text, on edge rows.

Tiers: exact (1e-10) for the ANOVA against scipy on the same float64 values, the
isotonic means and statistics; close for BMDExpress's ANOVA, within ten times the
spread that float32 storage gives scipy's (relative, set before the comparison) plus
1e-15 absolute, BMDExpress's p being 1 - F(x) in double precision; calibrated for
p-values from permutations: within z SDs plus 2 / (B + 1), z the two-sided 0.001 / n
quantile of the n genes compared, so that a row fails by chance with probability 0.001;
identical for gene sets, verdicts, choices on BMDExpress's own fits and acceptance;
close for maxima, BMDExpress's at most 1e-4 above CytePool's, and for BMDs where both
programs choose the same model and reach the same maximum, within 10 sqrt(2 max(d,
5e-7)) standard errors, (BMDU - BMDL) / 3.29, d the gap of the log-likelihoods, where
a BMD that the likelihood does not pin down is checked on BMDExpress's printed curve;
information for agreement on CytePool's own fits, beside FastBMD's 76%. On datasets
simulated without responsive genes by ``simulate_dose_series``, the share of cell
types that the ANOVA pre-modelling test passes (information), and on a vehicle-only
design, that the Williams plus 1.5-fold prefilter alone passes far more (control).

Requirements:
    Python 3.13 with numpy, numba and scipy, run from the repository root, and access to
    ftp.ncbi.nlm.nih.gov for the first run. BMDExpress 2.30.520 (``BMDEXPRESS``), a
    Python with pybmds 25.2 (``BMDS_PYTHON``) and R 4.4 with jsonlite (``RSCRIPT``). The
    data directory is --data, else GSE147072_DATA, else cytepool-validation/GSE147072
    under XDG_CACHE_HOME or ~/.cache; it must be outside the repository.

Usage:
    PYTHONPATH=. python validation/toxicology_rule_sets.py
    PYTHONPATH=. python validation/toxicology_rule_sets.py --write-fixtures
    PYTHONPATH=. python validation/toxicology_rule_sets.py --benchmark
"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SERIES = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147072/"
FILES = {  # path under SERIES: bytes and SHA-256, as downloaded on 2026-11-21
    "suppl/GSE147072_SRA_20_Chemical_matrix.txt.gz": (
        5_728_043,
        "9da191f15dd7f56e88345c0a1ccbc3dfb7a9c0fe6396d0d5b955f82b24def594",
    ),
    "matrix/GSE147072_series_matrix.txt.gz": (
        60_807,
        "26f1d85c1fa9ed92c2a64c5f28b5f47d565ccc573112fc571269c24cf4b5b50f",
    ),
}
# GEO's name of each chemical, and the doses (mg/kg) and liver samples per dose that GEO
# records, the vehicle first.
CHEMICALS = {
    "de71": (
        "DE-71",
        (0.0, 0.38, 0.75, 1.5, 3.0, 15.0, 50.0, 100.0, 200.0, 500.0),
        (3, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    ),
    "acrylamide": (
        "Acrylamide",
        (0.0, 0.075, 0.156, 0.3125, 0.625, 1.25, 2.5, 5.0, 10.0),
        (4,) * 9,
    ),
    "milk_thistle": (
        "Milk Thistle Extract",
        (0.0, 39.1, 78.125, 156.25, 312.5, 625.0, 937.5, 1250.0, 1750.0),
        (4,) * 9,
    ),
}
N_PROBES = 2654
RULES = ("ntp_2018", "etap_2024")
BMR_SD = {"ntp_2018": 1.0, "etap_2024": 1.349}
PERMUTATIONS = 10_000
SEED = 20261122
SUBSET = 100  # the prefiltered probes of each chemical that BMDExpress models
FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "toxicology_rule_sets"
)
FIXTURE_PROBES = 100  # acrylamide's first probes, for the tests of Williams' test
BMD_CLOSE = 1e-4  # BMDS 2 converges to a relative 1e-8 in the likelihood
FIT_TOLERANCE = 1e-3  # log-likelihoods that agree to BMDExpress's printing and input


def require(condition: bool, fact: str) -> None:
    """Stop with a message when a fact about the data does not hold."""
    if not condition:
        sys.exit(f"GSE147072 does not match what this script expects: {fact}")


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(directory: Path) -> None:
    """Download the files missing from directory, keeping a download only if its
    SHA-256 is right, and check the SHA-256 of the files already there."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in FILES.items():
        path = directory / Path(name).name
        if path.exists():
            found = sha256(path)
            require(found == checksum, f"{path} has SHA-256 {found}, not {checksum}")
            continue
        print(f"downloading {SERIES}{name} ({size:,} bytes) to {directory}", flush=True)
        partial = directory / f"{path.name}.part"
        with urllib.request.urlopen(SERIES + name, timeout=120) as response:
            with partial.open("wb") as file:
                shutil.copyfileobj(response, file, 1 << 20)
        found = sha256(partial)
        if found != checksum:
            partial.unlink()
        require(found == checksum, f"the download of {name} has SHA-256 {found}")
        partial.replace(path)


@dataclass(frozen=True)
class Chemical:
    """One chemical's liver samples: ids, doses and counts (samples x probes)."""

    key: str
    name: str
    samples: list[str]
    doses: np.ndarray
    probes: list[str]
    counts: np.ndarray


def read_chemicals(directory: Path) -> list[Chemical]:
    """The three chemicals' liver samples, joined to the counts by the code that ends
    each sample's title, never by position."""
    characteristics: list[list[str]] = []
    rows: dict[str, list[str]] = {}
    for line in gzip.open(directory / "GSE147072_series_matrix.txt.gz", "rt"):
        if not line.startswith("!Sample_"):
            continue
        key, *values = line.rstrip("\n").split("\t")
        values = [value.strip('"') for value in values]
        if key == "!Sample_characteristics_ch1":
            characteristics.append(values)
        else:
            rows.setdefault(key, values)
    samples = []
    for i, title in enumerate(rows["!Sample_title"]):
        fields = dict(c[i].split(": ", 1) for c in characteristics)
        samples.append(
            {
                "code": title.split()[-1],
                "chemical": fields["chemical"],
                "tissue": fields["tissue"],
                "dose": float(fields["dose(mg/kg)"]),
            }
        )
    text = gzip.open(directory / "GSE147072_SRA_20_Chemical_matrix.txt.gz", "rt").read()
    table = [line.split("\t") for line in text.splitlines() if line.strip()]
    start = next(i for i, row in enumerate(table) if row[0] == "ID_REF")
    codes = table[start][1:]
    probes = [row[0] for row in table[start + 1 :]]
    require(len(probes) == N_PROBES, f"{N_PROBES} probes, got {len(probes)}")
    require(len(set(codes)) == len(codes), "each sample code once in the counts")
    counts = np.array([[int(v) for v in row[1:]] for row in table[start + 1 :]])
    column = {code: j for j, code in enumerate(codes)}
    found = []
    for key, (name, levels, sizes) in CHEMICALS.items():
        mine = sorted(
            (s for s in samples if s["chemical"] == name and s["tissue"] == "liver"),
            key=lambda s: (s["dose"], s["code"]),
        )
        doses = np.array([s["dose"] for s in mine])
        observed = tuple(int(np.sum(doses == level)) for level in levels)
        require(observed == sizes, f"{name}: liver samples per dose {sizes}")
        require(all(s["code"] in column for s in mine), f"{name}: counts of each")
        found.append(
            Chemical(
                key,
                name,
                [s["code"] for s in mine],
                doses,
                probes,
                counts[:, [column[s["code"]] for s in mine]].T.astype(np.float64),
            )
        )
    return found


def etap_values(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ETAP Sec. 3.4.5.2, as the reference reads it: probes whose mean count is at
    least 5 are kept, each sample's counts per million over them (count x 1,000,000 /
    the sum of the kept probes' counts), then log2(CPM + 1)."""
    keep = counts.mean(axis=0) >= 5.0
    kept = counts[:, keep]
    return keep, np.log2(kept * 1e6 / kept.sum(axis=1, keepdims=True) + 1.0)


def stored(values: np.ndarray) -> np.ndarray:
    """The values as BMDExpress stores them, float32, in float64."""
    return values.astype(np.float32).astype(np.float64)


# ---- Seeded genes built to push each model's parameters outward --------------------

PUSHED_DOSES = np.repeat([0.0, 0.3, 1.0, 3.0, 10.0, 30.0], 4)


def pushed_genes() -> tuple[list[str], np.ndarray]:
    """Genes whose curves push the models' parameters to their bounds: lines (Hill's k
    without an upper bound, exponential 4's c), steps at the lowest dose (k at 0, large
    rates), steep and shallow rises (n at 18 and 1), concave and late rises, lines whose
    BMD lies far beyond the doses, and flat genes; log2 units, SD 0.1 or 0.15."""
    rng = np.random.default_rng(SEED)
    x = PUSHED_DOSES
    hill = lambda top, k, n: top * x**n / (k**n + x**n)  # noqa: E731
    shapes: dict[str, Any] = {}
    for sign, tag in ((1, "up"), (-1, "down")):
        shapes |= {
            f"line_{tag}": sign * 0.05 * x,
            f"convex_{tag}": sign * 0.0015 * x**2,
            f"step_lowest_{tag}": sign * 1.5 * (x > 0),
            f"jump_{tag}": sign * 1.5 * (x >= 10),
            f"shallow_{tag}": sign * 0.35 * np.log10(1.0 + x / 0.001),
            f"square_root_{tag}": sign * 0.27 * np.sqrt(x),
            f"late_{tag}": sign * 1.5 * (x / 30.0) ** 25,
            f"saturating_{tag}": sign * 1.5 * (1.0 - np.exp(-20.0 * x)),
            f"steep_{tag}": sign * hill(1.5, 5.0, 12.0),
            f"concave_{tag}": sign * (0.12 * x - 0.0035 * x**2),
            f"hill_{tag}": sign * hill(1.5, 3.0, 2.0),
        }
    for i in range(4):
        shapes[f"near_linear_{i}"] = 0.03 * x + 0.0004 * x**2
    for multiple in (5, 20, 100, 1000):
        shapes[f"far_line_{multiple}x"] = 1.349 * 0.1 / (multiple * 30.0) * x
    for i in range(3):
        shapes[f"flat_{i}"] = 0.0 * x
    names = list(shapes)
    sd = np.array([0.15 if n.startswith("near_linear") else 0.1 for n in names])
    values = 8.0 + np.column_stack([shapes[n] for n in names])
    values = values + rng.normal(0.0, 1.0, values.shape) * sd
    return names, values


# ---- BMDExpress 2.30.520 from its command line --------------------------------------

BMDEXPRESS_MODELS = [
    {"@type": "hill"},
    {"@type": "power"},
    {"@type": "poly", "degree": 1},
    {"@type": "poly", "degree": 2},
    {"@type": "exp", "expModel": 2},
    {"@type": "exp", "expModel": 3},
    {"@type": "exp", "expModel": 4},
    {"@type": "exp", "expModel": 5},
]
# Each rule set's settings in BMDExpress: the model choice, the BMR in SDs, and the gene
# filters of a category analysis, which apply the document's acceptance.
BMDEXPRESS_RULES = {
    "etap_2024": (
        {
            "bestPolyTest": 2,
            "pValueCutoff": 0.05,
            "flagHillWithKParameter": True,
            "kParameterValue": 3,
            "bestModelSelectionWithFlaggedHill": 2,
            "modifyFlaggedHillWithFractionMinBMD": 0.5,
        },
        {
            "removeBMDGreaterHighDose": True,
            "bmdPValueCutoff": 0.1,
            "bmdBMDLRatioMin": 20.0,
        },
    ),
    "ntp_2018": (
        {
            "bestPolyTest": 2,
            "pValueCutoff": 0.05,
            "flagHillWithKParameter": False,
            "kParameterValue": 1,
            "bestModelSelectionWithFlaggedHill": 1,
            "modifyFlaggedHillWithFractionMinBMD": 0.5,
        },
        {
            "removeBMDGreaterHighDose": True,
            "bmdPValueCutoff": 0.1,
            "bmduBMDLRatioMin": 40.0,
            "nFoldBelowLowestDose": 10.0,
        },
    ),
}
KILL_TIME = 60  # seconds; one exponential fit of the design's probe hung for 600


def bmdexpress_text(
    samples: list[str], doses: np.ndarray, probes: list[str], values: np.ndarray
) -> str:
    """BMDExpress's expression file: samples, doses, then one row per probe of the
    values as float32, the precision BMDExpress stores them in."""
    lines = [
        "Samples\t" + "\t".join(samples),
        "Probe set\t" + "\t".join(f"{d:g}" for d in doses),
    ]
    for j, probe in enumerate(probes):
        lines.append(probe + "\t" + "\t".join(str(np.float32(v)) for v in values[:, j]))
    return "\n".join(lines) + "\n"


def export_rows(path: Path) -> list[dict[str, str]]:
    """The rows of one of BMDExpress's tab-separated exports."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if line.startswith("Analysis\t"))
    header = lines[start].split("\t")
    return [
        dict(zip(header, line.split("\t"), strict=False)) for line in lines[start + 1 :]
    ]


def number(text: str | None) -> float:
    """BMDExpress's number; NaN for a blank, "NaN" and its failed fit's -9999."""
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return math.nan if value == -9999.0 else value


def by_probe(rows: list[dict[str, str]], analysis: str) -> dict[str, dict[str, str]]:
    return {row["Probe ID"]: row for row in rows if row["Analysis"] == analysis}


def _analyze(command: str, root: Path, config: dict[str, Any], name: str) -> float:
    """One run of ``analyze``, stopping with its log's end when it fails."""
    path = root / f"{name}.json"
    path.write_text(json.dumps(config))
    log = root / f"{name}.log"
    start = time.perf_counter()
    with log.open("w") as out:
        done = subprocess.run(
            [command, "analyze", "--config-file", str(path)],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    text = log.read_text()
    if done.returncode != 0 or "Exception in thread" in text or "Error:" in text:
        sys.exit("BMDExpress's analyze failed:\n" + "\n".join(text.splitlines()[-30:]))
    return time.perf_counter() - start


def _export(command: str, bm2: Path, group: str, path: Path) -> list[dict[str, str]]:
    subprocess.run(
        [command, "export", "--analysis-group", group, "--input-bm2", str(bm2)]
        + ["--output-file-name", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return export_rows(path)


def run_bmdexpress(
    data: dict[str, tuple[list[str], np.ndarray, list[str], np.ndarray]],
    pushed: tuple[list[str], np.ndarray],
) -> dict[str, Any]:
    """BMDExpress's prefilters of every probe of each chemical, then its BMD analyses
    under each rule set's settings of the first ``SUBSET`` probes its prefilter keeps
    and of the pushed genes, with the gene filters."""
    command = os.environ.get("BMDEXPRESS", "bmdexpress2-cmd")
    threads = os.cpu_count() or 2
    found: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("base", "work"):
            (root / name).mkdir()
        common = {"filterOutControlGenes": False, "numberOfThreads": threads}
        first: dict[str, Any] = {
            "bm2FileName": str(root / "prefilters.bm2"),
            "overwrite": True,
            "basePath": str(root / "base"),
            "expressionDataConfigs": [],
            "preFilterConfigs": [],
        }
        for key, (samples, doses, probes, values) in data.items():
            (root / f"{key}.txt").write_text(
                bmdexpress_text(samples, doses, probes, values)
            )
            first["expressionDataConfigs"].append(
                {
                    "inputFileName": str(root / f"{key}.txt"),
                    "outputName": key,
                    "logTransformation": "BASE2",
                    "platform": "generic",
                }
            )
            williams = common | {
                "@type": "williams",
                "inputName": key,
                "numberOfPermutations": PERMUTATIONS,
                "useMultipleTestingCorrection": False,
            }
            first["preFilterConfigs"] += [
                common
                | {
                    "@type": "anova",
                    "inputName": key,
                    "outputName": f"{key}_anova",
                    "pValueCutoff": 1.0,
                    "useMultipleTestingCorrection": True,
                    "useFoldChange": False,
                    "foldChange": 1.0,
                },
                williams
                | {
                    "outputName": f"{key}_williams",
                    "pValueCutoff": 1.0,
                    "useFoldChange": False,
                    "foldChange": 1.0,
                },
                williams
                | {
                    "outputName": f"{key}_prefilter",
                    "pValueCutoff": 0.05,
                    "useFoldChange": True,
                    "foldChange": 1.5,
                },
            ]
        seconds = _analyze(command, root, first, "prefilters")
        for group in ("anova", "williams"):
            found[group] = _export(
                command, root / "prefilters.bm2", group, root / f"{group}.txt"
            )
        # The second run: BMD analyses of each chemical's first prefiltered probes, and
        # of the pushed genes.
        subsets: dict[str, list[str]] = {}
        second: dict[str, Any] = {
            "bm2FileName": str(root / "bmds.bm2"),
            "overwrite": True,
            "basePath": str(root / "base"),
            "expressionDataConfigs": [],
            "bmdsConfigs": [],
            "categoryAnalysisConfigs": [],
        }
        sets = dict(data) | {
            "pushed": (
                [f"s{i}" for i in range(PUSHED_DOSES.size)],
                PUSHED_DOSES,
                pushed[0],
                pushed[1],
            )
        }
        for key, (samples, doses, probes, values) in sets.items():
            if key == "pushed":
                chosen = probes
            else:
                kept = by_probe(found["williams"], f"{key}_prefilter")
                chosen = [p for p in probes if p in kept][:SUBSET]
            subsets[key] = chosen
            columns = [probes.index(p) for p in chosen]
            path = root / f"{key}_subset.txt"
            path.write_text(bmdexpress_text(samples, doses, chosen, values[:, columns]))
            second["expressionDataConfigs"].append(
                {
                    "inputFileName": str(path),
                    "outputName": f"{key}_subset",
                    "logTransformation": "BASE2",
                    "platform": "generic",
                }
            )
            (root / f"{key}_probes.txt").write_text(
                "Probe\tGene\n"
                + "".join(f"{p}\t{100000 + i}\n" for i, p in enumerate(chosen))
            )
            (root / f"{key}_categories.txt").write_text(
                "Category ID\tCategory Name\tComponent\n"
                + "".join(f"C{i}\t{p}\t{100000 + i}\n" for i, p in enumerate(chosen))
            )
            for rule, (selection, filters) in BMDEXPRESS_RULES.items():
                second["bmdsConfigs"].append(
                    {
                        "modelConfigs": BMDEXPRESS_MODELS,
                        "bmdsBestModelSelection": selection,
                        "bmdsInputConfig": {
                            "maxIterations": 250,
                            "confidenceLevel": 0.95,
                            "constantVariance": True,
                            "restrictPower": True,
                            "bmrFactor": BMR_SD[rule],
                        },
                        "inputCategory": "expression",
                        "inputName": f"{key}_subset",
                        "outputName": f"{key}_{rule}",
                        "numberOfThreads": threads,
                        "killTime": KILL_TIME,
                        "tmpFolder": str(root / "work"),
                    }
                )
                second["categoryAnalysisConfigs"].append(
                    {
                        "@type": "defined",
                        "inputName": f"{key}_{rule}",
                        "outputName": f"{key}_{rule}_genes",
                        "removePromiscuousProbes": False,
                        "removeBMDGreaterHighDose": False,
                        "identifyConflictingProbeSets": False,
                        "correlationCutoffForConflictingProbeSets": 0.5,
                        "probeFilePath": str(root / f"{key}_probes.txt"),
                        "categoryFilePath": str(root / f"{key}_categories.txt"),
                        "deduplicateGeneSets": False,
                    }
                    | filters
                )
        seconds += _analyze(command, root, second, "bmds")
        found["bmd"] = _export(command, root / "bmds.bm2", "bmd", root / "bmd.txt")
        found["categorical"] = _export(
            command, root / "bmds.bm2", "categorical", root / "categorical.txt"
        )
        found["subsets"] = subsets
        found["seconds"] = seconds
        version = subprocess.run(
            [command, "--version"], capture_output=True, text=True, check=False
        )
        found["version"] = (version.stdout or version.stderr).strip().splitlines()[-1:]
    return found


# ---- pybmds 25.2: BMDS's recommended model ------------------------------------------

PYBMDS_MODELS = {
    "linear": "Linear",
    "polynomial_2": "Polynomial",
    "power": "Power",
    "hill": "Hill",
    "exponential_3": "ExponentialM3",
    "exponential_5": "ExponentialM5",
}


def _pybmds_gene(task: dict[str, Any]) -> dict[str, Any]:
    """pybmds's six continuous models of one gene, normal errors of constant variance,
    at a BMR of 1 SD, with its recommender's bins and choice."""
    import pybmds

    session = pybmds.Session(
        dataset=pybmds.ContinuousIndividualDataset(
            doses=task["doses"], responses=task["values"]
        )
    )
    for model in PYBMDS_MODELS:
        settings = {
            "disttype": pybmds.ContinuousDistType.normal,
            "bmr_type": pybmds.ContinuousRiskType.StandardDeviation,
            "bmr": 1.0,
        }
        if model == "polynomial_2":
            settings["degree"] = 2
        session.add_model(
            getattr(pybmds.Models, PYBMDS_MODELS[model]), settings=settings
        )
    session.execute()
    session.recommend()
    recommendation = session.recommender.results
    models = []
    for i, fitted in enumerate(session.models):
        result = fitted.results.model_dump()
        bounded = result["parameters"].get("bounded") or []
        models.append(
            {
                "log_likelihood": result["fit"]["loglikelihood"],
                "aic": result["fit"]["aic"],
                "n_bounded": int(sum(bool(b) for b in bounded[:-1])),
                "bmd": result["bmd"],
                "bmdl": result["bmdl"],
                "bin": int(recommendation.bmds_model_bin[i]),
            }
        )
    return {
        "key": task["key"],
        "models": models,
        "recommended": recommendation.recommended_model_index,
    }


def pybmds_main(directory: Path) -> None:
    """pybmds's fits of every gene of the input, in parallel processes."""
    from concurrent.futures import ProcessPoolExecutor
    from importlib.metadata import version

    tasks = json.loads((directory / "pybmds_input.json").read_text())
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = list(pool.map(_pybmds_gene, tasks, chunksize=4))
    (directory / "pybmds.json").write_text(
        json.dumps({"versions": {"pybmds": version("pybmds")}, "results": results})
    )


def run_pybmds(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """This script's pybmds part, under BMDS_PYTHON and isolated from the working tree.
    pybmds prints NLopt's message that its wheel lacks the Luksan algorithms once per
    optimizer call; the count is kept instead of the lines."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "pybmds_input.json").write_text(json.dumps(tasks))
        done = subprocess.run(
            [os.environ.get("BMDS_PYTHON", "python"), "-I", __file__, "--pybmds", tmp],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        found: dict[str, Any] = json.loads((directory / "pybmds.json").read_text())
    found["luksan_messages"] = done.stdout.count("Luksan code disabled")
    return found


# ---- R: both documents' acceptance criteria, coded from their text ------------------

R_ACCEPTANCE = r"""
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
number <- function(x) as.numeric(x)  # "Inf" and "NaN" arrive as text
bmd <- number(input$bmd); bmdl <- number(input$bmdl); bmdu <- number(input$bmdu)
p <- number(input$fit_p); top <- input$top; lowest <- input$lowest
known <- function(x) !is.na(x) & is.finite(x)
# NTP 2018, p. 9: convergent BMD, BMDL and BMDU (a BMDL of 0 or an unbounded BMDU is
# not), BMD below the highest dose, fit p-value above 0.1, BMDU / BMDL below 40; p. 8:
# a BMD more than 10-fold below the lowest positive dose is flagged and kept.
ntp <- known(bmd) & known(bmdl) & bmdl > 0 & known(bmdu) & bmd < top & p > 0.1 &
  bmdu / bmdl < 40
ntp_flag <- known(bmd) & bmd < lowest / 10
# US EPA 2024, Sec. 3.4.5.3: BMD below the highest dose, fit p-value above 0.1, BMD /
# BMDL below 20 (a BMDL of 0 makes the ratio infinite).
etap <- known(bmd) & !is.na(bmdl) & bmd < top & p > 0.1 & bmd / bmdl < 20
ntp[is.na(ntp)] <- FALSE
etap[is.na(etap)] <- FALSE
write_json(list(R = as.character(getRversion()), ntp = ntp, ntp_flag = ntp_flag,
                etap = etap), args[[2]], auto_unbox = TRUE)
"""
# The edge rows: each criterion at its threshold and beside it, with 0, infinite and
# missing bounds; the doses are those of the pushed genes (lowest 0.3, top 30).
EDGE_ROWS = [
    # bmd, bmdl, bmdu, fit p-value
    (1.0, 0.5, 2.0, 0.5),
    (30.0, 15.0, 60.0, 0.5),
    (np.nextafter(30.0, 0.0), 15.0, 60.0, 0.5),
    (1.0, 0.5, 2.0, 0.1),
    (1.0, 0.5, 2.0, np.nextafter(0.1, 1.0)),
    (2.0, 0.1, 3.0, 0.5),
    (np.nextafter(2.0, 0.0), 0.1, 3.0, 0.5),
    (1.0, 0.1, 4.0, 0.5),
    (1.0, 0.1, np.nextafter(4.0, 0.0), 0.5),
    (0.03, 0.01, 0.05, 0.5),
    (np.nextafter(0.03, 0.0), 0.01, 0.05, 0.5),
    (1.0, 0.0, 2.0, 0.5),
    (1.0, 0.5, math.inf, 0.5),
    (1.0, math.nan, 2.0, 0.5),
    (math.nan, math.nan, math.nan, math.nan),
    (1.0, 0.5, 2.0, math.nan),
]


def run_r(program: str, payload: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "input.json").write_text(json.dumps(payload))
        (root / "program.R").write_text(program)
        subprocess.run(
            [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", str(root / "program.R")]
            + [str(root / "input.json"), str(root / "output.json")],
            check=True,
        )
        found: dict[str, Any] = json.loads((root / "output.json").read_text())
    return found


def text_numbers(values: Any) -> list[str]:
    """Numbers as text for R: repr keeps every digit, and "Inf" and "NaN" read back."""
    return [
        repr(float(v)) if math.isfinite(v) else ("NaN" if math.isnan(v) else "Inf")
        for v in values
    ]


# ---- Exact p-values of Williams' trend test, over every relabelling -----------------

SMALL_DESIGNS = {"4 x 2": (2, 2, 2, 2), "3, 2, 2, 2": (3, 2, 2, 2), "3 x 3": (3, 3, 3)}
SMALL_PERMUTATIONS = 20_000
SMALL_GENES = 60


def small_design(sizes: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Doses and values of seeded genes of one small design: a quarter without a
    trend, the others rising, falling or peaking by 0.5 to 2 SDs."""
    rng = np.random.default_rng(seed)
    k = len(sizes)
    doses = np.repeat(np.arange(k, dtype=float) ** 2 / (k - 1) ** 2 * 10.0, sizes)
    level = np.repeat(np.arange(k), sizes)
    shapes = [
        np.zeros(k),
        np.linspace(0.0, 1.0, k),
        -np.linspace(0.0, 1.0, k),
        np.sin(np.linspace(0.0, np.pi, k)),
    ]
    columns = [
        rng.normal(size=level.size) + shapes[g % 4][level] * rng.uniform(0.5, 2.0)
        for g in range(SMALL_GENES)
    ]
    return doses, np.round(np.column_stack(columns), 6)


def end_statistics(groups: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """t_up and t_down for every row of group labels, by the minimax formula of
    isotonic regression (the rising fit is min_j mean(0..j) at the vehicle and max_i
    mean(i..K-1) at the top dose; the falling fit swaps them), not by
    pool-adjacent-violators."""
    one_hot = groups[:, :, None] == np.arange(k)[None, None, :]
    n = one_hot.sum(axis=1).astype(float)
    centred = y - y.mean()
    sums = np.einsum("ank,n->ak", one_hot, centred)
    within = (centred**2).sum() - (sums**2 / n).sum(axis=1)
    scale = np.sqrt(np.maximum(within, 0.0) / (y.size - k)) * np.sqrt(
        1.0 / n[:, -1] + 1.0 / n[:, 0]
    )
    head = np.cumsum(sums, axis=1) / np.cumsum(n, axis=1)
    tail = (np.cumsum(sums[:, ::-1], axis=1) / np.cumsum(n[:, ::-1], axis=1))[:, ::-1]
    rise = tail.max(axis=1) - head.min(axis=1)
    fall = head.max(axis=1) - tail.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(rise > 0, rise / scale, 0.0)
        down = np.where(fall > 0, fall / scale, 0.0)
    return np.column_stack([up, down])


def exact_p_values(doses: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    """For every gene, over every distinct relabelling: the p-value conditional on the
    gene's side, the share of relabellings on that side, and whether it is up."""
    levels = np.unique(doses)
    labels = np.searchsorted(levels, doses)
    table = np.array(sorted(set(itertools.permutations(labels.tolist()))))
    conditional, share, up = [], [], []
    for g in range(values.shape[1]):
        y = values[:, g]
        observed = end_statistics(labels[None, :], y, levels.size)[0]
        upward = bool(observed[0] >= observed[1])
        every = end_statistics(table, y, levels.size)
        same = every[:, 0] >= every[:, 1] if upward else every[:, 1] > every[:, 0]
        reach = (every[:, 0] if upward else every[:, 1]) >= observed.max() * (
            1.0 - 1e-9
        )
        conditional.append(np.count_nonzero(same & reach) / np.count_nonzero(same))
        share.append(np.count_nonzero(same) / len(table))
        up.append(upward)
    return {
        "p_value": np.array(conditional),
        "share": np.array(share),
        "up": np.array(up),
    }


# ---- CytePool -----------------------------------------------------------------------


@dataclass
class Ours:
    """What CytePool computed: each chemical's rule sets in dose_response, Williams'
    trend test on BMDExpress's values, the subset's curves and BMDs under each rule set,
    BMDS's recommendation of pybmds's genes and the small designs' tests."""

    results: dict[tuple[str, str], Any] = field(default_factory=dict)
    williams: dict[str, Any] = field(default_factory=dict)
    subset: dict[tuple[str, str], tuple[list[str], Any, Any]] = field(
        default_factory=dict
    )
    recommended: Any = None
    small: dict[str, Any] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    subset_data: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def chemical_cube(chemical: Chemical) -> Any:
    from cytepool.cube.pool import from_bulk
    from cytepool.table import Table

    samples = Table({"sample_id": chemical.samples, "dose": chemical.doses})
    return from_bulk(
        chemical.counts.astype(np.int64),
        samples=samples,
        features=Table({"feature_id": chemical.probes}),
    )


def run_cytepool(
    chemicals: list[Chemical],
    values: dict[str, tuple[list[str], np.ndarray]],
    subsets: dict[str, list[str]],
    pushed: tuple[list[str], np.ndarray],
    recommendation_sets: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
) -> Ours:
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.design import Design
    from cytepool.dose_axes import DoseAxis
    from cytepool.dose_responses import dose_response
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses
    from cytepool.stats.trend_tests import williams_trend_test

    axis = DoseAxis("dose", unit="mg/kg", metric="administered")
    ours = Ours()
    for chemical in chemicals:
        cube = chemical_cube(chemical)
        for rule in RULES:
            start = time.perf_counter()
            ours.results[chemical.key, rule] = dose_response(
                cube, Design(cube.samples, "~ 1"), dose=axis, rules=rule, rng=SEED
            )
            ours.seconds[f"dose_response {chemical.key} {rule}"] = (
                time.perf_counter() - start
            )
        probes, kept = values[chemical.key]
        ours.williams[chemical.key] = williams_trend_test(
            chemical.doses, stored(kept), n_permutations=PERMUTATIONS, rng=SEED
        )
    sets = {
        c.key: (c.doses, values[c.key][0], stored(values[c.key][1])) for c in chemicals
    }
    sets["pushed"] = (PUSHED_DOSES, pushed[0], stored(pushed[1]))
    for key, (doses, probes, kept) in sets.items():
        ids = subsets[key]
        columns = [probes.index(p) for p in ids]
        fits = fit_dose_curves(doses, kept[:, columns], models="bmdexpress")
        ours.subset_data[key] = (doses, kept[:, columns])
        for rule in RULES:
            found = benchmark_doses(
                fits, BenchmarkResponse.sd(BMR_SD[rule]), model=rule
            )
            ours.subset[key, rule] = (ids, fits, found)
    ours.recommended = {}
    for name, (doses, _, genes) in recommendation_sets.items():
        epa = fit_dose_curves(doses, genes, models="epa")
        ours.recommended[name] = (
            epa,
            benchmark_doses(epa, BenchmarkResponse.sd(1.0), model="bmds_2026"),
        )
    for i, (name, sizes) in enumerate(SMALL_DESIGNS.items()):
        small_doses, small_values = small_design(sizes, SEED + i)
        ours.small[name] = williams_trend_test(
            small_doses, small_values, n_permutations=SMALL_PERMUTATIONS, rng=SEED + i
        )
    return ours


def bmds_sets(
    chemicals: list[Chemical],
    values: dict[str, tuple[list[str], np.ndarray]],
    subsets: dict[str, list[str]],
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """pybmds's genes, by design: the acrylamide and milk thistle probes of the subset,
    and 60 seeded genes of shapes BMDS recommends among (lines, Hill and exponential
    rises, plateaus, steps and flat genes) at the doses of the pushed genes."""
    sets = {}
    for chemical in chemicals:
        if chemical.key != "de71":
            probes, kept = values[chemical.key]
            columns = [probes.index(p) for p in subsets[chemical.key]]
            sets[chemical.key] = (
                chemical.doses,
                subsets[chemical.key],
                stored(kept[:, columns]),
            )
    rng = np.random.default_rng(SEED + 10)
    x = PUSHED_DOSES
    columns = []
    for g in range(60):
        top = rng.uniform(0.3, 1.5) * (1 if g % 2 else -1)
        shape = [
            top * x / 30,
            top * x / (rng.uniform(1, 10) + x),
            top * (1 - np.exp(-x / rng.uniform(1, 10))),
            top * (x >= 10),
            0 * x,
        ][g % 5]
        columns.append(8.0 + shape + rng.normal(0.0, 0.1, x.size))
    sets["seeded"] = (
        x,
        [f"seeded_{g}" for g in range(60)],
        stored(np.column_stack(columns)),
    )
    return sets


def bmds_tasks(
    sets: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
) -> list[dict[str, Any]]:
    return [
        {
            "key": f"{name}/{gene}",
            "doses": doses.tolist(),
            "values": values[:, j].tolist(),
        }
        for name, (doses, genes, values) in sets.items()
        for j, gene in enumerate(genes)
    ]


# ---- The comparisons ----------------------------------------------------------------


@dataclass
class Rows:
    """The rows of the printed table, and how many failed."""

    rows: list[tuple[str, ...]] = field(default_factory=list)
    failed: int = 0

    def bounded(
        self, check: str, reference: str, value: float, bound: float, text: str = ""
    ) -> None:
        passed = bool(value <= bound)
        self.failed += not passed
        shown = text or f"{value:.1e}"
        self.rows.append(
            (check, reference, shown, f"{bound:g}", "pass" if passed else "FAIL")
        )

    def identical(self, check: str, reference: str, same: int, total: int) -> None:
        passed = same == total
        self.failed += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{same:,} of {total:,} identical",
                "identical",
                "pass" if passed else "FAIL",
            )
        )

    def control(
        self, check: str, reference: str, value: float, bound: float, text: str
    ) -> None:
        passed = bool(value > bound)
        self.failed += not passed
        self.rows.append(
            (check, reference, text, f"> {bound:.3g}", "control" if passed else "FAIL")
        )

    def information(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "info"))

    def deliberate(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "deliberate difference"))


NAMES = {
    "linear": "Linear",
    "polynomial_2": "Poly 2",
    "power": "Power",
    "hill": "Hill",
    "exponential_2": "Exp 2",
    "exponential_3": "Exp 3",
    "exponential_4": "Exp 4",
    "exponential_5": "Exp 5",
}
CHEMICAL_NAMES = {key: name for key, (name, _, _) in CHEMICALS.items()}


def quantile(n: int) -> float:
    """z of the two-sided 0.001 / n tail of the standard normal: a calibrated row of n
    comparisons fails by chance with probability 0.001."""
    from scipy.stats import norm

    return float(norm.isf(0.0005 / n))


def compare_anova(
    chemicals: list[Chemical],
    values: dict[str, Any],
    ours: Ours,
    theirs: dict[str, Any],
    rows: Rows,
) -> None:
    from scipy.stats import f_oneway, false_discovery_control

    exact, close, passing, verdicts = 0.0, 0.0, [0, 0], [0, 0]
    adjusted_close = 0.0
    spreads = []
    for chemical in chemicals:
        probes, kept = values[chemical.key]
        table = ours.results[chemical.key, "etap_2024"].table("bulk", "rule_set")
        assert table["feature_id"].tolist() == probes
        levels = np.unique(chemical.doses)
        wide = f_oneway(*[kept[chemical.doses == level] for level in levels], axis=0)
        exact = max(
            exact,
            float(np.max(np.abs(table["anova_p_value"] - wide.pvalue) / wide.pvalue)),
        )
        # BMDExpress's values are float32: ten times the spread they give scipy's F, p
        # and adjusted p, fixed before the comparison, plus BMDExpress's 1 - F(x).
        narrow = f_oneway(
            *[stored(kept)[chemical.doses == level] for level in levels], axis=0
        )
        p_spread = 10.0 * float(
            np.max(np.abs(narrow.pvalue - wide.pvalue) / wide.pvalue)
        )
        adjusted_wide = false_discovery_control(wide.pvalue)
        adjusted_narrow = false_discovery_control(narrow.pvalue)
        a_spread = 10.0 * float(
            np.max(np.abs(adjusted_narrow - adjusted_wide) / adjusted_wide)
        )
        spreads.append((p_spread, a_spread))
        anova = by_probe(theirs["anova"], f"{chemical.key}_anova")
        p_b = np.array([number(anova[p]["Unadjusted P-Value"]) for p in probes])
        a_b = np.array([number(anova[p]["Adjusted P-Value"]) for p in probes])
        ours_p = np.asarray(table["anova_p_value"])
        ours_a = np.asarray(table["anova_adjusted_p_value"])
        close = max(
            close, float(np.max(np.abs(ours_p - p_b) / (p_spread * ours_p + 1e-15)))
        )
        # Benjamini-Hochberg multiplies each p-value by up to n / rank, and with it
        # the granularity of BMDExpress's 1 - F(x).
        adjusted_close = max(
            adjusted_close,
            float(
                np.max(np.abs(ours_a - a_b) / (a_spread * ours_a + len(probes) * 1e-15))
            ),
        )
        mine, yours = ours_a < 0.05, a_b < 0.05
        passing[0] += int(np.sum(mine == yours))
        passing[1] += len(probes)
        verdicts[0] += int(mine.any() == yours.any())
        verdicts[1] += 1
    rows.bounded(
        "ANOVA p-values, the rule table against scipy's f_oneway on the same values",
        "scipy 1.18.1",
        exact,
        1e-10,
    )
    bound = ", ".join(f"{s[0]:.1e}" for s in spreads)
    rows.bounded(
        "ANOVA p-values against BMDExpress, over 10 x float32's spread ("
        + bound
        + ") x p + 1e-15",
        "BMDExpress 2.30.520",
        close,
        1.0,
    )
    rows.bounded(
        "Benjamini-Hochberg adjusted p-values against BMDExpress, the same bound with n x 1e-15",
        "BMDExpress 2.30.520",
        adjusted_close,
        1.0,
    )
    rows.identical(
        "genes passing the ANOVA pre-modelling test (adjusted p < 0.05)",
        "BMDExpress 2.30.520",
        *passing,
    )
    rows.identical(
        "each chemical's verdict, analysed or inactive",
        "BMDExpress 2.30.520",
        *verdicts,
    )
    for chemical in chemicals:
        record = ours.results[chemical.key, "etap_2024"].provenance["groups"][0][
            "rule_set"
        ]
        rows.information(
            f"{CHEMICAL_NAMES[chemical.key]}: the ANOVA pre-modelling test",
            "dose_response",
            f"{record['n_genes_passing_anova']:,} of {record['n_genes']:,} genes; {record['status']}",
        )


def compare_williams(
    chemicals: list[Chemical],
    values: dict[str, Any],
    ours: Ours,
    theirs: dict[str, Any],
    rows: Rows,
) -> None:
    from scipy.optimize import isotonic_regression

    means_gap = stat_gap = 0.0
    p_rows = []
    fold, fold_verdicts, prefilter = 0.0, [0, 0], [0, 0]
    for chemical in chemicals:
        probes, kept = values[chemical.key]
        y = stored(kept)
        test = ours.williams[chemical.key]
        levels = np.unique(chemical.doses)
        labels = np.searchsorted(levels, chemical.doses)
        sizes = np.bincount(labels).astype(float)
        group_means = np.array(
            [y[labels == k].mean(axis=0) for k in range(levels.size)]
        ).T
        within = np.array(
            [
                ((y[labels == k] - y[labels == k].mean(axis=0)) ** 2).sum(axis=0)
                for k in range(levels.size)
            ]
        ).sum(axis=0)
        s = np.sqrt(within / (y.shape[0] - levels.size))
        factor = np.sqrt(1 / sizes[-1] + 1 / sizes[0])
        for g in range(len(probes)):
            up = isotonic_regression(group_means[g], weights=sizes, increasing=True).x
            down = isotonic_regression(
                group_means[g], weights=sizes, increasing=False
            ).x
            t_up = max(up[-1] - up[0], 0.0) / (s[g] * factor)
            t_down = max(down[0] - down[-1], 0.0) / (s[g] * factor)
            fit = up if test.direction[g] == "up" else down
            means_gap = max(
                means_gap,
                float(
                    np.max(
                        np.abs(test.isotonic_means[g] - fit)
                        / np.maximum(np.abs(fit), 1.0)
                    )
                ),
            )
            stat_gap = max(
                stat_gap,
                abs(test.statistic[g] - max(t_up, t_down)) / max(t_up, t_down, 1.0),
            )
        williams = by_probe(theirs["williams"], f"{chemical.key}_williams")
        p_b = np.array([number(williams[p]["Unadjusted P-Value"]) for p in probes])
        p_rows.append((test.p_value, p_b))
        changes = group_means[:, 1:] - group_means[:, :1]
        largest = changes[np.arange(len(probes)), np.argmax(np.abs(changes), axis=1)]
        fc_b = np.array([number(williams[p]["Max Fold Change"]) for p in probes])
        log2_b = np.sign(fc_b) * np.log2(np.abs(fc_b))
        fold = max(fold, float(np.max(np.abs(largest - log2_b))))
        near = np.abs(np.abs(largest) - np.log2(1.5)) < 1e-5
        fold_verdicts[0] += int(
            np.sum(
                (np.abs(largest) >= np.log2(1.5))[~near]
                == (np.abs(log2_b) >= np.log2(1.5))[~near]
            )
        )
        fold_verdicts[1] += int(np.sum(~near))
        kept_b = by_probe(theirs["williams"], f"{chemical.key}_prefilter")
        theirs_pass = np.array([p in kept_b for p in probes])
        mine = (test.p_value < 0.05) & (np.abs(largest) >= np.log2(1.5))
        prefilter[0] += int(np.sum(mine == theirs_pass))
        prefilter[1] += len(probes)
    rows.bounded(
        "isotonic means against scipy's isotonic_regression (BMDExpress's values)",
        "scipy 1.18.1",
        means_gap,
        1e-10,
    )
    rows.bounded(
        "Williams' statistics against the formula from scipy's isotonic fits",
        "scipy 1.18.1",
        stat_gap,
        1e-10,
    )
    total = sum(p.size for p, _ in p_rows)
    z = quantile(total)
    worst, beyond = 0.0, 0
    for p_c, p_b in p_rows:
        mean = (p_c + p_b) / 2
        # Each p-value counts the relabellings on its side, at least 0.45 of B here.
        sd = np.sqrt(mean * (1 - mean) * 2 / (0.45 * PERMUTATIONS))
        gap = np.abs(p_c - p_b)
        worst = max(worst, float(np.max(gap / (z * sd + 2 / (PERMUTATIONS + 1)))))
        beyond += int(np.sum(gap > 4 * sd + 2 / (PERMUTATIONS + 1)))
    rows.bounded(
        f"Williams p-values against BMDExpress's, B = {PERMUTATIONS:,} each: the largest gap over z SDs + 2 / (B + 1), z = {z:.2f} for {total:,} genes; {beyond} beyond 4 SDs",
        "BMDExpress 2.30.520",
        worst,
        1.0,
        f"{worst:.3f}",
    )
    small_worst, small_beyond, small_total, sides = 0.0, 0, 0, [0, 0]
    n_small = SMALL_GENES * len(SMALL_DESIGNS)
    z_small = quantile(n_small)
    for i, (name, sizes) in enumerate(SMALL_DESIGNS.items()):
        doses, genes = small_design(sizes, SEED + i)
        exact = exact_p_values(doses, genes)
        test = ours.small[name]
        sd = np.sqrt(
            exact["p_value"]
            * (1 - exact["p_value"])
            / (exact["share"] * SMALL_PERMUTATIONS)
        )
        gap = np.abs(test.p_value - exact["p_value"])
        small_worst = max(
            small_worst,
            float(np.max(gap / (z_small * sd + 2 / (SMALL_PERMUTATIONS + 1)))),
        )
        small_beyond += int(np.sum(gap > 4 * sd + 2 / (SMALL_PERMUTATIONS + 1)))
        sides[0] += int(np.sum((test.direction == "up") == exact["up"]))
        sides[1] += genes.shape[1]
        small_total += genes.shape[1]
    rows.bounded(
        f"Williams p-values on seeded small designs ({', '.join(SMALL_DESIGNS)}) against exact enumeration of every relabelling, B = {SMALL_PERMUTATIONS:,}: largest gap over z SDs + 2 / (B + 1), z = {z_small:.2f}; {small_beyond} beyond 4 SDs",
        "enumeration",
        small_worst,
        1.0,
        f"{small_worst:.3f}",
    )
    rows.identical(
        "the direction of Williams' test on the small designs", "enumeration", *sides
    )
    rows.bounded(
        "largest change from the vehicle, log2, against BMDExpress's fold change (float32)",
        "BMDExpress 2.30.520",
        fold,
        1e-5,
    )
    rows.identical(
        "fold-change verdicts, at least 1.5-fold (none within 1e-5 of it left out)",
        "BMDExpress 2.30.520",
        *fold_verdicts,
    )
    rows.information(
        "prefilter calls, Williams p < 0.05 and 1.5-fold (each program's own permutations)",
        "BMDExpress 2.30.520",
        f"{prefilter[0]:,} of {prefilter[1]:,} agree ({prefilter[0] / prefilter[1]:.2%})",
    )


def their_mean(model: str, row: dict[str, str], x: np.ndarray) -> np.ndarray:
    """BMDExpress's curve of a model at doses x, from its printed parameters, by BMDS
    2's formulas."""
    label = NAMES[model]

    def q(name: str) -> float:
        return number(row[f"{label} Parameter {name}"])

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if model == "linear":
            return q("beta_0") + q("beta_1") * x
        if model == "polynomial_2":
            return q("beta_0") + q("beta_1") * x + q("beta_2") * x**2
        if model == "power":
            return q("control") + q("slope") * x ** q("power")
        if model == "hill":
            n = q("n")
            return q("Intercept") + q("v") * x**n / (q("k") ** n + x**n)
        if model == "exponential_2":
            return q("a") * np.exp(q("sign") * q("b") * x)
        if model == "exponential_3":
            return q("a") * np.exp(q("sign") * (q("b") * x) ** q("d"))
        if model == "exponential_4":
            return q("a") * (q("c") - (q("c") - 1.0) * np.exp(-q("b") * x))
        return q("a") * (q("c") - (q("c") - 1.0) * np.exp(-((q("b") * x) ** q("d"))))


def their_curve_at_maximum(
    model: str, row: dict[str, str], doses: np.ndarray, y: np.ndarray, bmr_sd: float
) -> tuple[float, float]:
    """BMDExpress's printed curve on a gene's samples: its log-likelihood under
    constant variance, and the BMD it gives, the lowest dose up to 50 times the top at
    which its change from dose 0 reaches bmr_sd times its maximum-likelihood SD (F6)."""
    from scipy.optimize import brentq

    n = y.size
    rss = float(np.sum((y - their_mean(model, row, doses)) ** 2))
    sd = math.sqrt(rss / n)
    likelihood = -0.5 * n * (math.log(2 * math.pi * rss / n) + 1.0)
    level = float(their_mean(model, row, np.array([0.0]))[0])

    def excess(x: float) -> float:
        change = abs(float(their_mean(model, row, np.array([x]))[0]) - level)
        return change - bmr_sd * sd

    top = float(doses.max())
    grid = np.geomspace(top * 1e-6, 50 * top, 4000)
    values = np.array([excess(x) for x in grid])
    above = np.flatnonzero(values >= 0)
    if above.size == 0:
        return likelihood, math.nan
    j = int(above[0])
    if j == 0:
        return likelihood, float(grid[0])
    return likelihood, float(brentq(excess, grid[j - 1], grid[j], xtol=1e-12 * top))


def model_fits_agree(fits: Any, i: int, row: dict[str, str], n: int) -> dict[str, bool]:
    """For each model, whether CytePool's fit and BMDExpress's reach the same maximum
    (log-likelihoods within FIT_TOLERANCE, BMDExpress's without its n/2 log(2 pi))."""
    shift = n / 2 * math.log(2 * math.pi)
    out = {}
    for name, label in NAMES.items():
        m = fits.models.index(name)
        theirs = number(row[f"{label} fitLogLikelihood"]) - shift
        out[name] = bool(
            math.isfinite(theirs)
            and abs(fits.log_likelihood[i, m] - theirs) < FIT_TOLERANCE
        )
    return out


def compare_fits_and_choices(
    chemicals: list[Chemical], ours: Ours, theirs: dict[str, Any], rows: Rows
) -> None:
    from cytepool.stats.potency import _bmds2_counted, _first_with_bounds

    doses_of = {c.key: c.doses for c in chemicals} | {"pushed": PUSHED_DOSES}
    shortfall, capped, n_fits = 0.0, [], 0
    counts = [0, 0]
    on_their_fits, same, wide_bmdls = [0, 0], {rule: [0, 0] for rule in RULES}, [0, 0]
    bmd_gap, bound_gap, n_bmds, floored = 0.0, 0.0, 0, 0
    # Bounds compared; at the ends of CytePool's search where BMDS 2's lie beyond; within
    # 5e-3; at those ends where BMDS 2's lie within the search.
    bounds_seen = [0, 0, 0, 0]
    bmd_relative: list[float] = []
    ridges: list[float] = []
    largest_gap: tuple[float, str] = (0.0, "")
    for (key, rule), (ids, fits, found) in ours.subset.items():
        n = doses_of[key].size
        table = by_probe(theirs["bmd"], f"{key}_{rule}")
        counted = _bmds2_counted(fits)
        lowest = float(np.float32(doses_of[key][doses_of[key] > 0][0]))
        their_aic = np.array(
            [[number(table[p][f"{NAMES[m]} AIC"]) for m in NAMES] for p in ids]
        )
        their_bmd = np.array(
            [[number(table[p][f"{NAMES[m]} BMD"]) for m in NAMES] for p in ids]
        )
        their_bmdl = np.array(
            [[number(table[p][f"{NAMES[m]} BMDL"]) for m in NAMES] for p in ids]
        )
        their_bmdu = np.array(
            [[number(table[p][f"{NAMES[m]} BMDU"]) for m in NAMES] for p in ids]
        )
        k = np.array([number(table[p]["Hill Parameter k"]) for p in ids])
        open_ = np.isfinite(their_aic) & np.isfinite(their_bmd)
        if rule == "etap_2024":
            open_[:, list(NAMES).index("hill")] &= ~(k < lowest / 3)
        chosen = _first_with_bounds(
            their_aic,
            open_,
            lambda g, m: ~np.isnan(their_bmdl[g, m]) & ~np.isnan(their_bmdu[g, m]),
        )
        best = [table[p]["Best Model"] for p in ids]
        rebuilt = [NAMES[list(NAMES)[c]] if c >= 0 else "" for c in chosen]
        on_their_fits[0] += sum(a == b for a, b in zip(rebuilt, best))
        on_their_fits[1] += len(ids)
        if rule != "etap_2024":
            # Each fit's maximum, once per subset (the fits do not depend on the BMR).
            for i, p in enumerate(ids):
                for name, label in NAMES.items():
                    m = fits.models.index(name)
                    theirs_ll = number(
                        table[p][f"{label} fitLogLikelihood"]
                    ) - n / 2 * math.log(2 * math.pi)
                    if not math.isfinite(theirs_ll):
                        continue
                    n_fits += 1
                    gap = theirs_ll - fits.log_likelihood[i, m]
                    c = (
                        fits.parameters[i, m, fits.parameter_names.index("c")]
                        if name in ("exponential_4", "exponential_5")
                        else math.nan
                    )
                    if c == 1000.0:
                        if gap > 0:
                            capped.append(gap)
                        continue
                    shortfall = max(shortfall, gap)
        for i, p in enumerate(ids):
            agree = model_fits_agree(fits, i, table[p], n)
            for name, label in NAMES.items():
                m = fits.models.index(name)
                if not agree[name]:
                    continue
                # BMDS 2's AIC penalty, (AIC + 2 LL) / 2, from printed values.
                theirs_count = (
                    round(
                        (
                            number(table[p][f"{label} AIC"])
                            + 2 * number(table[p][f"{label} fitLogLikelihood"])
                        )
                        / 2
                    )
                    - 1
                )
                counts[0] += int(theirs_count == counted[i, m])
                counts[1] += 1
            theirs_model = {v: k for k, v in NAMES.items()}.get(
                table[p]["Best Model"], ""
            )
            same[rule][0] += int(found.gene_models[i] == theirs_model)
            same[rule][1] += 1
            if found.gene_models[i] == theirs_model and agree.get(theirs_model, False):
                b = number(table[p]["Best BMD"])
                if found.status[0, i] == "below a hundredth of the lowest dose":
                    floored += 1
                elif math.isfinite(b) and math.isfinite(found.bmd[0, i]):
                    # A log-likelihood difference d at the maximum moves a parameter by
                    # about sqrt(2 d) of its standard errors; the BMD's standard error
                    # is about (BMDU - BMDL) / 3.29, from its two-sided 90% interval.
                    m = fits.models.index(theirs_model)
                    shift = n / 2 * math.log(2 * math.pi)
                    d = abs(
                        fits.log_likelihood[i, m]
                        - (
                            number(table[p][f"{NAMES[theirs_model]} fitLogLikelihood"])
                            - shift
                        )
                    )
                    width = number(table[p]["Best BMDU"]) - number(
                        table[p]["Best BMDL"]
                    )
                    allowed = 10 * math.sqrt(2 * max(d, 5e-7)) * width / 3.29
                    gap = abs(found.bmd[0, i] - b) / allowed
                    ridge = False
                    if gap > 1.0:
                        # Where the likelihood is flat along the BMD, as for a step
                        # between two doses, curves of one maximum cross the BMR at
                        # different doses: BMDExpress's own printed curve then reaches
                        # CytePool's maximum and gives its printed BMD.
                        subset_doses, data = ours.subset_data[key]
                        ll, crossing = their_curve_at_maximum(
                            theirs_model,
                            table[p],
                            subset_doses,
                            data[:, i],
                            BMR_SD[rule],
                        )
                        ridge = (
                            abs(ll - fits.log_likelihood[i, m]) < FIT_TOLERANCE
                            and abs(crossing - b) <= 1e-4 * b
                        )
                    if ridge:
                        ridges.append(found.bmd[0, i] / b)
                    else:
                        n_bmds += 1
                        bmd_gap = max(bmd_gap, gap)
                        bmd_relative.append(abs(found.bmd[0, i] - b) / b)
                        largest_gap = max(
                            largest_gap, (gap, f"{key} {rule} {p} {theirs_model}")
                        )
                    lowest_dose = float(doses_of[key][doses_of[key] > 0][0])
                    top_dose = float(doses_of[key].max())
                    for ours_bound, label in (
                        (found.bmdl[0, i], "Best BMDL"),
                        (found.bmdu[0, i], "Best BMDU"),
                    ):
                        theirs_bound = number(table[p][label])
                        if not (math.isfinite(theirs_bound) and theirs_bound > 0):
                            continue
                        # CytePool's search ends at a hundredth of the lowest dose and
                        # 50 times the top dose, where BMDS 2 goes on.
                        beyond = (
                            ours_bound == 0.0 and theirs_bound < lowest_dose / 100
                        ) or (ours_bound == math.inf and theirs_bound > 50 * top_dose)
                        gap = abs(ours_bound - theirs_bound) / theirs_bound
                        bounds_seen[0] += 1
                        if beyond:
                            bounds_seen[1] += 1
                        elif gap <= 5e-3:
                            bounds_seen[2] += 1
                        elif ours_bound in (0.0, math.inf):
                            bounds_seen[3] += 1
                        else:
                            bound_gap = max(
                                bound_gap,
                                ours_bound / theirs_bound,
                                theirs_bound / ours_bound,
                            )
            # The genes whose qualifying models' BMDLs span more than 3-fold: both rule
            # sets keep the model of lowest AIC, as BMDExpress does.
            qualifying = open_[i] & ~np.isnan(their_bmdl[i]) & (their_bmdl[i] > 0)
            spread = their_bmdl[i][qualifying]
            if spread.size > 1 and spread.max() / spread.min() > 3:
                wide_bmdls[0] += int(rebuilt[i] == best[i])
                wide_bmdls[1] += 1
    rows.bounded(
        f"each model's maximum against BMDExpress's, {n_fits:,} fits but those with c at CytePool's limit: BMDExpress's above CytePool's by at most",
        "BMDExpress 2.30.520",
        shortfall,
        1e-4,
    )
    rows.deliberate(
        "exponential 4 and 5 with c at CytePool's limit of 1,000, where BMDS 2's c exceeds it",
        "BMDExpress 2.30.520",
        f"{len(capped)} fits, BMDExpress higher by at most {max(capped, default=0.0):.3f}",
    )
    rows.information(
        "parameters counted in the AIC, where the maxima agree: the others lie on flat ridges of the likelihood, where BMDS 2's optimizer stops short of a bound or past it",
        "BMDExpress 2.30.520",
        f"{counts[0]:,} of {counts[1]:,} identical",
    )
    rows.identical(
        "chosen model, CytePool's rule on BMDExpress's own fits, AICs and bounds",
        "BMDExpress 2.30.520",
        *on_their_fits,
    )
    rows.identical(
        "genes whose qualifying BMDLs span more than 3-fold keep the lowest AIC",
        "BMDExpress 2.30.520",
        *wide_bmdls,
    )
    for rule in RULES:
        found_same, total = same[rule]
        rows.information(
            f"chosen model on CytePool's own fits, {rule} (FastBMD's agreement with BMDExpress: 76%)",
            "BMDExpress 2.30.520",
            f"{found_same:,} of {total:,} ({found_same / total:.1%})",
        )
    rows.bounded(
        f"BMD where the models and their maxima agree ({n_bmds:,} genes): the largest gap over 10 sqrt(2 max(d, 5e-7)) (BMDU - BMDL) / 3.29, d the log-likelihood difference ({largest_gap[1]}); median relative gap {np.median(bmd_relative):.1e}",
        "BMDExpress 2.30.520",
        bmd_gap,
        1.0,
        f"{bmd_gap:.3f}",
    )
    rows.information(
        "BMDs that the data do not pin down: BMDExpress's printed curve reaches CytePool's maximum within 1e-3 and crosses the BMR at its printed BMD, where CytePool's curve of the same likelihood crosses elsewhere",
        "BMDExpress 2.30.520",
        f"{len(ridges)} genes, CytePool's BMD {min(ridges, default=1.0):.3f} to {max(ridges, default=1.0):.3f} times BMDExpress's",
    )
    compared, at_ends, close, short = bounds_seen
    rows.information(
        "BMDLs and BMDUs of the BMDs compared, BMDS 2's and CytePool's both the ends of the piece of the profile's region that holds the BMD",
        "BMDExpress 2.30.520",
        f"{close:,} of {compared:,} within 5e-3; {at_ends:,} at the ends of CytePool's search where BMDS 2's lie beyond them; {short:,} at those ends where BMDS 2's lie within the search; the other {compared - close - at_ends - short:,} within a factor of {bound_gap:.2f}",
    )
    rows.deliberate(
        "BMDs below a hundredth of the lowest dose, which CytePool gives as that dose and BMDS 2 extrapolates",
        "BMDExpress 2.30.520",
        f"{floored} genes",
    )


def acceptance_inputs(
    bmd: Any, bmdl: Any, bmdu: Any, fit_p: Any, lowest: float, top: float
) -> Any:
    """What the documents' checks read of BMDs, bounds and fit p-values given as
    numbers."""
    from cytepool.stats._benchmark_dose_flags import FlagInputs

    n = len(bmd)
    missing = np.full(n, np.nan)
    return FlagInputs(
        bmd=np.asarray(bmd, dtype=float),
        bmdl=np.asarray(bmdl, dtype=float),
        bmdu=np.asarray(bmdu, dtype=float),
        bounds=True,
        lowest_dose=lowest,
        top_dose=top,
        fit_p_value=np.asarray(fit_p, dtype=float),
        residual_near_bmd=missing,
        residual_at_vehicle=missing,
        vehicle_sd_ratio=missing,
        constant_variance_p_value=missing,
    )


def verdicts(rule: str, inputs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Each row's acceptance by CytePool's rule of the document, and whether it is
    flagged and kept."""
    from cytepool.stats._benchmark_dose_flags import accepted_rows, failed_checks

    flagged = np.array(
        ["(NTP keeps the gene)" in text for text in failed_checks(rule, inputs)]
    )
    return accepted_rows(rule, inputs), flagged


def compare_acceptance(
    chemicals: list[Chemical],
    ours: Ours,
    theirs: dict[str, Any],
    r: dict[str, Any],
    rows: Rows,
) -> None:
    doses_of = {c.key: c.doses for c in chemicals} | {"pushed": PUSHED_DOSES}
    categories = {
        (row["Analysis"], row["GO/Pathway/Gene Set/Gene Name"]): row
        for row in theirs["categorical"]
    }
    same = total = printed = 0
    own = {rule: [0, 0] for rule in RULES}
    for (key, rule), (ids, _, found) in ours.subset.items():
        table = by_probe(theirs["bmd"], f"{key}_{rule}")
        doses = doses_of[key]
        best = {
            c: np.array([number(table[p][f"Best {c}"]) for p in ids])
            for c in ("BMD", "BMDL", "BMDU", "fitPValue")
        }
        accepted, flagged = verdicts(
            rule,
            acceptance_inputs(
                best["BMD"],
                best["BMDL"],
                best["BMDU"],
                best["fitPValue"],
                float(np.float32(doses[doses > 0][0])),
                float(np.float32(doses.max())),
            ),
        )
        # BMDExpress's NTP filters remove the BMDs that NTP flags and keeps.
        expected = accepted & ~flagged if rule == "ntp_2018" else accepted
        kept = np.array(
            [
                categories.get((f"{key}_{rule}_genes", p), {}).get(
                    "Genes That Passed All Filters"
                )
                == "1"
                for p in ids
            ]
        )
        # BMDS 2 prints fit p-values to 4 decimals: at exactly 0.1 BMDExpress's filter
        # keeps a gene that the documents' "above 0.1" does not.
        tie = best["fitPValue"] == 0.1
        same += int(np.sum((expected == kept) & ~tie))
        total += int(np.sum(~tie))
        printed += int(np.sum(tie & (expected != kept)))
        # CytePool's own BMDs, bounds and test 4 where both programs chose one model.
        names = {label: name for name, label in NAMES.items()}
        chosen = np.array([names.get(table[p]["Best Model"], "") for p in ids])
        agree = (found.gene_models == chosen) & (chosen != "")
        mine = found.accepted(rule)[0]
        if rule == "ntp_2018":
            mine = mine & ~np.array(
                [
                    "10-fold below" in t
                    for t in found.table(flags=rule)[f"flags_{rule}"].tolist()[
                        : len(ids)
                    ]
                ]
            )
        own[rule][0] += int(np.sum((mine == kept) & agree & ~tie))
        own[rule][1] += int(np.sum(agree & ~tie))
    rows.identical(
        "acceptance on BMDExpress's BMDs, bounds and fit p-values against its own gene filters",
        "BMDExpress 2.30.520",
        same,
        total,
    )
    rows.information(
        "fit p-values printed as exactly 0.1, which BMDExpress's filter keeps and the documents do not",
        "BMDExpress 2.30.520",
        f"{printed} genes",
    )
    for rule in RULES:
        rows.information(
            f"{rule}'s acceptance on CytePool's own BMDs, bounds and test 4, where both programs chose one model, against BMDExpress's filters",
            "BMDExpress 2.30.520",
            f"{own[rule][0]:,} of {own[rule][1]:,} identical",
        )
    edges = np.array(EDGE_ROWS, dtype=float)
    for rule in RULES:
        accepted, flagged = verdicts(
            rule,
            acceptance_inputs(
                edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3], 0.3, 30.0
            ),
        )
        coded = np.array(r["etap" if rule == "etap_2024" else "ntp"], dtype=bool)
        rows.identical(
            f"{rule}'s acceptance on {len(EDGE_ROWS)} edge rows (each criterion at its threshold)",
            "R, from the documents' text",
            int(np.sum(accepted == coded)),
            len(EDGE_ROWS),
        )
        if rule == "ntp_2018":
            rows.identical(
                "NTP's flag of a BMD more than 10-fold below the lowest dose, kept",
                "R, from NTP 2018",
                int(np.sum(flagged == np.array(r["ntp_flag"], dtype=bool))),
                len(EDGE_ROWS),
            )
    # A control on the BMD / BMDL threshold: ETAP's accepted genes of DE-71 change when
    # the threshold of 20 becomes 10 or 40.
    table = ours.results["de71", "etap_2024"].table("bulk", "rule_set")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(table["bmd"]) / np.asarray(table["bmdl"])
    other = np.asarray(table["accepted"])
    tighter = int(np.sum(other & (ratio >= 10)))
    passing = (
        np.asarray(table["passes_prefilter"])
        & np.isfinite(ratio)
        & (table["failed_checks"] == "BMD / BMDL >= 20")
    )
    looser = int(np.sum(passing & (ratio < 40)))
    rows.control(
        "DE-71 under ETAP: accepted genes that BMD / BMDL < 10 would reject",
        "the rule set",
        tighter,
        0,
        f"{tighter} genes",
    )
    rows.control(
        "DE-71 under ETAP: genes rejected only for BMD / BMDL >= 20 that < 40 would accept",
        "the rule set",
        looser,
        0,
        f"{looser} genes",
    )


def compare_recommendation(
    sets: dict[str, Any], ours: Ours, pybmds: dict[str, Any], rows: Rows
) -> None:
    """BMDS's recommended model against pybmds's, identical where both programs reach
    the same maximum with the same parameters at a bound for every model."""
    from cytepool.stats.potency import VIABLE_AIC, _pybmds_counted

    results = {result["key"]: result for result in pybmds["results"]}
    order = [
        "linear",
        "polynomial_2",
        "power",
        "hill",
        "exponential_3",
        "exponential_5",
    ]
    same = total = differ = tied = 0
    for name, (doses, genes, _) in sets.items():
        fits, found = ours.recommended[name]
        counted = _pybmds_counted(fits)
        aic = -2.0 * fits.log_likelihood + 2.0 * (counted + 1.0)
        for i, gene in enumerate(genes):
            theirs = results[f"{name}/{gene}"]
            agree = True
            for j, model in enumerate(order):
                m = fits.models.index(model)
                row = theirs["models"][j]
                sizes = len(
                    [
                        q
                        for q in fits.parameter_names
                        if not np.isnan(
                            fits.parameters[i, m, fits.parameter_names.index(q)]
                        )
                    ]
                )
                if not (
                    abs(fits.log_likelihood[i, m] - row["log_likelihood"])
                    < FIT_TOLERANCE
                    and sizes - row["n_bounded"] == counted[i, m]
                ):
                    agree = False
            index = theirs["recommended"]
            recommended = order[index] if index is not None else ""
            if agree:
                total += 1
                ours_model = found.gene_models[i]
                if ours_model == recommended:
                    same += 1
                elif ours_model and recommended and found.model_choice[i] == VIABLE_AIC:
                    # Viable models whose AICs lie within the fits' agreement of the
                    # lowest tie in both programs; pybmds breaks such a tie by its
                    # optimizer's last digits, CytePool by the fewest parameters.
                    pair = [
                        fits.models.index(ours_model),
                        fits.models.index(recommended),
                    ]
                    theirs_aic = {
                        m: theirs["models"][j]["aic"]
                        for j, m in enumerate(order)
                        if m in (ours_model, recommended)
                    }
                    close = (
                        abs(aic[i, pair[0]] - aic[i, pair[1]]) < 2 * FIT_TOLERANCE
                        and abs(theirs_aic[ours_model] - theirs_aic[recommended])
                        < 2 * FIT_TOLERANCE
                    )
                    same += int(close)
                    tied += int(close)
            else:
                differ += int(found.gene_models[i] != recommended)
    rows.identical(
        "BMDS's recommended model where both programs agree on every model's maximum and "
        f"bounds ({tied} chosen by AIC among models tied within the fits' agreement, "
        "2e-3, which pybmds breaks by its optimizer's last digits and CytePool by the "
        "fewest parameters, count as identical)",
        "pybmds 25.2",
        same,
        total,
    )
    rows.information(
        "genes where a fit or its bounded parameters differ, and the recommendation with it",
        "pybmds 25.2",
        f"{differ} genes; pybmds printed NLopt's Luksan message {pybmds['luksan_messages']:,} times",
    )


def compare_minimums(chemicals: list[Chemical], ours: Ours, rows: Rows) -> None:
    for chemical in chemicals:
        result = ours.results[chemical.key, "etap_2024"]
        found = result.rules("bulk")
        outcome = dict(zip(found["setting"].tolist(), found["outcome"].tolist()))
        rows.information(
            f"{CHEMICAL_NAMES[chemical.key]}: ETAP's design minimums",
            "check_dose_design",
            outcome["design minimums"],
        )


# ---- Null and power datasets --------------------------------------------------------

NULL_DESIGNS = {
    "K = 4 x 3": ((0.0, 0.1, 1.0, 10.0), 3),
    "K = 9 x 2": ((0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), 2),
    "K = 9 x 3": ((0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), 3),
}
NULL_DATASETS = 200
VEHICLE_ONLY = ((0.0, 1.0, 2.0, 3.0), 6)  # Lee 2024: 4 mock groups of 6 samples
VEHICLE_DATASETS = 100
POWER_DATASETS = 10


def null_rows(rows: Rows) -> None:
    """The ANOVA pre-modelling test on datasets without responsive genes: the share of
    cell types it passes, which the documents' test does not promise for pseudobulk
    counts (information); on the vehicle-only design of Lee 2024, the Williams plus
    1.5-fold prefilter without it must pass more than 5% plus 4 binomial SEs (the
    control); on power datasets, the shares of responsive and flat genes accepted."""
    from cytepool.design import Design
    from cytepool.dose_axes import DoseAxis
    from cytepool.dose_responses import dose_response
    from cytepool.simulated_dose_series import simulate_dose_series
    from cytepool.stats.trend_tests import williams_trend_test

    axis = DoseAxis("dose", unit="ug/kg", metric="administered")
    for cell, (name, (doses, replicates)) in enumerate(NULL_DESIGNS.items()):
        passed = accepted = n = 0
        # The ANOVA's p-values below 1e-4, of the genes of mean count 5 to 10 and of
        # the others, against the 1e-4 of uniform p-values.
        small = {True: [0, 0], False: [0, 0]}
        for dataset in range(NULL_DATASETS):
            cube, _ = simulate_dose_series(
                doses,
                replicates=replicates,
                cell_types=2,
                n_genes=1000,
                responsive=0.0,
                rng=np.random.default_rng([SEED, 1, cell, dataset]),
            )
            result = dose_response(
                cube,
                Design(cube.samples, "~ 1"),
                dose=axis,
                rules="ntp_2018",
                rng=dataset,
            )
            summary = result.summary
            passed += int(np.sum(summary["n_genes_passing_anova"] > 0))
            accepted += int(np.sum(summary["n_genes_accepted"] > 0))
            n += len(summary)
            for g, group in enumerate(result.groups):
                table = result.table(group, "rule_set")
                counts = np.asarray(cube.layer("counts", g), dtype=np.float64)
                position = {f: i for i, f in enumerate(cube.features["feature_id"])}
                columns = [position[f] for f in table["feature_id"].tolist()]
                few = counts[:, columns].mean(axis=0) < 10.0
                low = np.asarray(table["anova_p_value"]) < 1e-4
                for side in (True, False):
                    small[side][0] += int(np.sum(low & (few == side)))
                    small[side][1] += int(np.sum(few == side))
        rows.information(
            f"null datasets, {name}: share of {n} cell types passing the ANOVA "
            "pre-modelling test, against its nominal 0.05",
            "the truth",
            f"{passed / n:.3f} (binomial SE at 0.05: {math.sqrt(0.05 * 0.95 / n):.3f})",
        )
        rows.information(
            f"null datasets, {name}: share of cell types with an accepted gene",
            "the truth",
            f"{accepted / n:.3f}",
        )
        rows.information(
            f"null datasets, {name}: share of the ANOVA's p-values below 1e-4, genes of mean count 5 to 10 and the others, against 1e-4",
            "the truth",
            f"{small[True][0] / small[True][1]:.1e} of {small[True][1]:,}; "
            f"{small[False][0] / small[False][1]:.1e} of {small[False][1]:,}",
        )
    doses, replicates = VEHICLE_ONLY
    for cells in (30.0, 500.0):
        prefilter = gate = 0
        for dataset in range(VEHICLE_DATASETS):
            cube, _ = simulate_dose_series(
                doses,
                replicates=replicates,
                cell_types=1,
                cells=cells,
                n_genes=1000,
                responsive=0.0,
                rng=np.random.default_rng([SEED, 4, int(cells), dataset]),
            )
            counts = np.asarray(cube.layer("counts", 0), dtype=float)
            sample_doses = np.asarray(cube.samples["dose"], dtype=float)
            _, values = etap_values(counts)
            test = williams_trend_test(
                sample_doses, values, n_permutations=1_000, rng=dataset
            )
            levels = np.unique(sample_doses)
            means = np.array(
                [values[sample_doses == level].mean(axis=0) for level in levels]
            )
            changes = np.abs(means[1:] - means[0]).max(axis=0)
            prefilter += int(np.any((test.p_value < 0.05) & (changes >= np.log2(1.5))))
            result = dose_response(
                cube,
                Design(cube.samples, "~ 1"),
                dose=axis,
                rules="ntp_2018",
                rng=dataset,
            )
            gate += int(result.summary["n_genes_passing_anova"][0] > 0)
        bound = 0.05 + 4 * math.sqrt(0.05 * 0.95 / VEHICLE_DATASETS)
        rows.control(
            f"vehicle only, 4 mock doses x 6 (Lee 2024), {cells:g} cells: Williams p < 0.05 and 1.5-fold without the ANOVA test, share of datasets with a gene",
            "the truth",
            prefilter / VEHICLE_DATASETS,
            bound,
            f"{prefilter / VEHICLE_DATASETS:.2f}",
        )
        rows.information(
            f"vehicle only, {cells:g} cells: share passing the ANOVA pre-modelling "
            "test, against its nominal 0.05",
            "the truth",
            f"{gate / VEHICLE_DATASETS:.2f} (binomial SE at 0.05: "
            f"{math.sqrt(0.05 * 0.95 / VEHICLE_DATASETS):.3f})",
        )
    responsive = flat = n_responsive = n_flat = 0
    doses, replicates = NULL_DESIGNS["K = 9 x 3"]
    for dataset in range(POWER_DATASETS):
        cube, truth = simulate_dose_series(
            doses,
            replicates=replicates,
            cell_types=2,
            n_genes=1000,
            rng=np.random.default_rng([SEED, 0, 2, dataset]),
        )
        result = dose_response(
            cube, Design(cube.samples, "~ 1"), dose=axis, rules="etap_2024", rng=dataset
        )
        for group in result.groups:
            known = truth.table(group)
            is_responsive = dict(
                zip(known["feature_id"].tolist(), (known["shape"] != "flat").tolist())
            )
            try:
                table = result.table(group, "rule_set")
            except KeyError:
                continue
            for gene, ok in zip(
                table["feature_id"].tolist(), table["accepted"].tolist()
            ):
                if is_responsive[gene]:
                    n_responsive += 1
                    responsive += int(ok)
                else:
                    n_flat += 1
                    flat += int(ok)
    rows.information(
        f"power datasets, K = 9 x 3, {POWER_DATASETS} x 2 cell types: genes ETAP accepts",
        "the truth",
        f"{responsive / max(n_responsive, 1):.3f} of responsive genes, {flat / max(n_flat, 1):.4f} of flat genes",
    )


# ---- Fixtures -----------------------------------------------------------------------


def write_fixtures(
    chemicals: list[Chemical],
    values: dict[str, Any],
    pushed: tuple[list[str], np.ndarray],
    theirs: dict[str, Any],
    r: dict[str, Any],
    sets: dict[str, Any],
    pybmds: dict[str, Any],
) -> None:
    """Small inputs and references for the tests: Williams p-values of acrylamide's
    first probes; BMDExpress's fits, choices and filters of the pushed genes and of 20
    probes of each chemical's subset; R's verdicts of the edge rows; pybmds's
    recommendations of its seeded genes."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    acrylamide = next(c for c in chemicals if c.key == "acrylamide")
    probes, kept = values["acrylamide"]
    williams = by_probe(theirs["williams"], "acrylamide_williams")
    first = probes[:FIXTURE_PROBES]
    (FIXTURE_DIRECTORY / "williams.json").write_text(
        json.dumps(
            {
                "doses": acrylamide.doses.tolist(),
                "probes": first,
                "values": stored(kept[:, :FIXTURE_PROBES]).tolist(),
                "permutations": PERMUTATIONS,
                "bmdexpress_p_values": [
                    number(williams[p]["Unadjusted P-Value"]) for p in first
                ],
            }
        )
    )
    genes: dict[str, Any] = {}
    sources = {
        c.key: (c.doses, values[c.key][0], stored(values[c.key][1])) for c in chemicals
    }
    sources["pushed"] = (PUSHED_DOSES, pushed[0], stored(pushed[1]))
    for key, (doses, names, matrix) in sources.items():
        chosen = (
            theirs["subsets"][key] if key == "pushed" else theirs["subsets"][key][:20]
        )
        entry: dict[str, Any] = {"doses": doses.tolist(), "probes": chosen}
        entry["values"] = matrix[:, [names.index(p) for p in chosen]].tolist()
        for rule in RULES:
            table = by_probe(theirs["bmd"], f"{key}_{rule}")
            categories = {
                row["GO/Pathway/Gene Set/Gene Name"]: row[
                    "Genes That Passed All Filters"
                ]
                == "1"
                for row in theirs["categorical"]
                if row["Analysis"] == f"{key}_{rule}_genes"
            }
            entry[rule] = (
                {
                    "best": [table[p]["Best Model"] for p in chosen],
                    "kept_by_filters": [categories.get(p, False) for p in chosen],
                    "hill_k": [number(table[p]["Hill Parameter k"]) for p in chosen],
                }
                | {
                    f"{field_} {model}": [
                        number(table[p][f"{NAMES[model]} {field_}"]) for p in chosen
                    ]
                    for model in NAMES
                    for field_ in ("fitLogLikelihood", "AIC", "BMD", "BMDL", "BMDU")
                }
                | {
                    f"Best {field_}": [
                        number(table[p][f"Best {field_}"]) for p in chosen
                    ]
                    for field_ in ("BMD", "BMDL", "BMDU", "fitPValue")
                }
            )
        genes[key] = entry
    (FIXTURE_DIRECTORY / "bmdexpress.json").write_text(json.dumps(genes))
    (FIXTURE_DIRECTORY / "acceptance.json").write_text(
        json.dumps(
            {"rows": np.array(EDGE_ROWS).tolist(), "lowest": 0.3, "top": 30.0}
            | {k: r[k] for k in ("ntp", "ntp_flag", "etap")}
        )
    )
    results = {result["key"]: result for result in pybmds["results"]}
    doses, names, matrix = sets["seeded"]
    (FIXTURE_DIRECTORY / "pybmds.json").write_text(
        json.dumps(
            {
                "doses": doses.tolist(),
                "genes": names,
                "values": matrix.tolist(),
                "recommended": [results[f"seeded/{g}"]["recommended"] for g in names],
                "log_likelihood": [
                    [m["log_likelihood"] for m in results[f"seeded/{g}"]["models"]]
                    for g in names
                ],
                "n_bounded": [
                    [m["n_bounded"] for m in results[f"seeded/{g}"]["models"]]
                    for g in names
                ],
            }
        )
    )
    for path in sorted(FIXTURE_DIRECTORY.iterdir()):
        print(
            f"wrote {path.relative_to(FIXTURE_DIRECTORY.parents[2])} ({path.stat().st_size:,} bytes)"
        )


# ---- Benchmark ----------------------------------------------------------------------


def benchmark() -> None:
    """williams_trend_test on 20,000 genes x 27 samples (the vehicle and 8 doses x 3)
    at 1,000 and 10,000 permutations, the median of three runs after a compiling call;
    dose_response(rules="etap_2024") on one cell type of 20,000 genes, and the BMDs at
    1 SD of 10,000 of its genes' "epa" curves by the best model and by BMDS's
    recommendation, one run after a first; each beside the weighted fit of 20,000
    genes x 200 samples of linear_models.py's benchmark."""
    from cytepool.design import Design
    from cytepool.dose_axes import DoseAxis
    from cytepool.dose_responses import dose_response
    from cytepool.simulated_dose_series import simulate_dose_series
    from cytepool.stats.linear_models import fit_linear_model
    from cytepool.stats.trend_tests import williams_trend_test

    def timed(run: Any, repeats: int = 3) -> float:
        run()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        return float(np.median(times))

    rng = np.random.default_rng(1)
    n_samples = 200
    groups = np.repeat(np.arange(4), n_samples // 4)
    x = np.column_stack(
        [np.ones(n_samples)]
        + [(groups == g).astype(float) for g in (1, 2, 3)]
        + [rng.normal(50.0, 10.0, n_samples), rng.normal(0.0, 1.0, n_samples)]
    )
    y = rng.normal(6.0, 1.0, (n_samples, 20_000))
    weights = np.exp(rng.normal(0.0, 0.5, y.shape))
    baseline = timed(lambda: fit_linear_model(y, x, weights=weights))
    print(
        f"Baseline: weighted fit_linear_model, 20,000 genes x 200 samples, {baseline:.3f} s"
    )
    doses = np.repeat([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], 3)
    values = rng.normal(8.0, 0.3, (doses.size, 20_000))
    print("| run | seconds | baselines | target |")
    print("| --- | --- | --- | --- |")
    for b in (1_000, 10_000):
        seconds = timed(
            lambda b=b: williams_trend_test(doses, values, n_permutations=b, rng=1)
        )
        print(
            f"| williams_trend_test, {b:,} permutations | {seconds:.2f} | {seconds / baseline:.0f} | 60 s at 1,000 permutations |"
        )
    cube, _ = simulate_dose_series(
        np.unique(doses), replicates=3, cell_types=1, cells=300.0, n_genes=20_000, rng=2
    )
    axis = DoseAxis("dose", unit="ug/kg", metric="administered")
    seconds = timed(
        lambda: dose_response(
            cube, Design(cube.samples, "~ 1"), dose=axis, rules="etap_2024", rng=1
        ),
        repeats=1,
    )
    print(
        f"| dose_response(rules='etap_2024'), one cell type of 20,000 genes | {seconds:.1f} | {seconds / baseline:.0f} | information |"
    )
    # BMDS's recommendation computes bounds for every model that its checks without
    # bounds leave viable, against the bounds of each gene's best model alone.
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    sample_doses = np.asarray(cube.samples["dose"], dtype=np.float64)
    _, log_values = etap_values(np.asarray(cube.layer("counts", 0), dtype=np.float64))
    fits = fit_dose_curves(sample_doses, log_values[:, :10_000], models="epa")
    for model in ("best", "bmds_2026"):
        seconds = timed(
            lambda model=model: benchmark_doses(
                fits, BenchmarkResponse.sd(1.0), model=model
            ),
            repeats=1,
        )
        print(
            f"| benchmark_doses(model={model!r}), 10,000 genes of the 'epa' set, with bounds | {seconds:.1f} | {seconds / baseline:.0f} | information |"
        )


# ---- Main ---------------------------------------------------------------------------


def peak_memory() -> float:
    """The largest resident memory of this process so far, in MiB; NaN off Linux."""
    import resource

    if sys.platform != "linux":
        return math.nan
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help="the directory of the GEO files")
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--benchmark", action="store_true", help="time the rule sets")
    parser.add_argument("--pybmds", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.pybmds is not None:
        pybmds_main(arguments.pybmds)
        return 0
    if arguments.benchmark:
        benchmark()
        return 0
    clock = time.perf_counter()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = Path(
        arguments.data
        or os.environ.get("GSE147072_DATA")
        or cache / "cytepool-validation" / "GSE147072"
    ).expanduser()
    if directory.resolve().is_relative_to(FIXTURE_DIRECTORY.parents[2]):
        sys.exit(f"{directory} is in the repository; keep the data outside it")
    fetch(directory)
    chemicals = read_chemicals(directory)
    values = {}
    for chemical in chemicals:
        keep, kept = etap_values(chemical.counts)
        values[chemical.key] = ([p for p, k in zip(chemical.probes, keep) if k], kept)
    pushed = pushed_genes()
    data = {
        c.key: (c.samples, c.doses, values[c.key][0], values[c.key][1])
        for c in chemicals
    }
    theirs = run_bmdexpress(data, pushed)
    print(
        f"BMDExpress: {theirs['version']}, analyze {theirs['seconds']:.0f} s; subsets "
        + ", ".join(f"{k} {len(v)}" for k, v in theirs["subsets"].items())
    )
    sets = bmds_sets(chemicals, values, theirs["subsets"])
    pybmds = run_pybmds(bmds_tasks(sets))
    edges = np.array(EDGE_ROWS, dtype=float)
    r = run_r(
        R_ACCEPTANCE,
        {
            "bmd": text_numbers(edges[:, 0]),
            "bmdl": text_numbers(edges[:, 1]),
            "bmdu": text_numbers(edges[:, 2]),
            "fit_p": text_numbers(edges[:, 3]),
            "lowest": 0.3,
            "top": 30.0,
        },
    )
    print(f"References: pybmds {pybmds['versions']['pybmds']}, R {r['R']}")
    start = time.perf_counter()
    ours = run_cytepool(chemicals, values, theirs["subsets"], pushed, sets)
    print(f"CytePool: {time.perf_counter() - start:.1f} s")
    rows = Rows()
    compare_anova(chemicals, values, ours, theirs, rows)
    compare_williams(chemicals, values, ours, theirs, rows)
    compare_fits_and_choices(chemicals, ours, theirs, rows)
    compare_acceptance(chemicals, ours, theirs, r, rows)
    compare_recommendation(sets, ours, pybmds, rows)
    compare_minimums(chemicals, ours, rows)
    null_rows(rows)
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows.rows:
        print("| " + " | ".join(row) + " |")
    if arguments.write_fixtures:
        if rows.failed:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(chemicals, values, pushed, theirs, r, sets, pybmds)
    print(
        f"{len(rows.rows)} checks, {rows.failed} failed; {time.perf_counter() - clock:.0f} s, peak memory {peak_memory():.0f} MiB"
    )
    return 1 if rows.failed else 0


if __name__ == "__main__":
    sys.exit(main())
