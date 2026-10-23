"""Validate adjust_p_values against R's p.adjust.

Each check runs every p.adjust method (holm, hochberg, hommel, bonferroni, BH,
BY, none) on one set of p-values and passes when the missing values agree and
the largest relative difference from R is at most 1e-10 (exact tier). Cases:
uniform p-values, many ties, missing values, exact 0 and 1, n larger than the
number of p-values, every vector of length 1 to 5 over a grid of values, the
columns of a matrix adjusted separately, random families, and 100,000
p-values. The cases labelled "fixture" are written to
tests/data/multiple_testing/ with --write-fixtures.

Requirements: Python 3.13+ with numpy and numba; R 4.4+ (base packages only).

Run from the repository root. R's Hommel procedure takes a few minutes on
100,000 p-values.

    RSCRIPT=/path/to/Rscript PYTHONPATH=. python validation/multiple_testing.py
    RSCRIPT=/path/to/Rscript PYTHONPATH=. python validation/multiple_testing.py \\
        --write-fixtures
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cytepool.stats.multiple_testing import adjust_p_values

METHODS = ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "none")
TOLERANCE = 1e-10
FIXTURES = Path(__file__).resolve().parents[1] / "tests/data/multiple_testing"
AXIS = {"vector": None, "rows": 1, "columns": 0}

# Arguments: the exchange directory, then the methods. Each case is a list of
# vectors stored back to back; each vector is adjusted as its own family.
R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
# n < 0 stands for the default, which leaves the missing values uncounted.
adjust <- function(p, method, n) {
  if (n < 0) p.adjust(p, method) else p.adjust(p, method, n = n)
}
cases <- read.table(file.path(args[1], "cases.tsv"), header = TRUE,
                    colClasses = c("character", "numeric"))
for (i in seq_len(nrow(cases))) {
  base <- file.path(args[1], cases$key[i])
  lengths <- scan(paste0(base, ".lengths"), quiet = TRUE)
  p <- readBin(paste0(base, ".p"), "double", sum(lengths), endian = "little")
  ends <- cumsum(lengths)
  for (method in args[-1]) {
    out <- numeric(length(p))
    for (j in seq_along(lengths)) {
      at <- seq_len(lengths[j]) + ends[j] - lengths[j]
      out[at] <- adjust(p[at], method, cases$n[i])
    }
    writeBin(out, paste0(base, ".", method), endian = "little")
  }
}
cat(R.version.string)
"""


def mixture(rng: np.random.Generator, size: int) -> np.ndarray:
    """Uniform null p-values with 15% non-null p-values piled up near 0."""
    p = rng.uniform(size=size)
    signal = rng.uniform(size=size) < 0.15
    p[signal] = rng.beta(0.1, 1.0, size=np.count_nonzero(signal))
    return p


def blank(rng: np.random.Generator, p: np.ndarray, fraction: float) -> np.ndarray:
    """Set a random fraction of the p-values to NaN."""
    p = p.copy()
    p[rng.uniform(size=p.shape) < fraction] = np.nan
    return p


class Case(NamedTuple):
    key: str
    label: str
    vectors: list[np.ndarray]  # each vector is one family of tests
    n: int = -1  # -1 is R's default, the number of non-missing p-values
    layout: str = "vector"  # how CytePool receives the vectors, see AXIS
    fixture: bool = False


def build_cases(rng: np.random.Generator) -> list[Case]:
    grid = (0.0, 0.01, 0.03, 0.5, 1.0, np.nan)
    small = [
        np.array(v) for k in range(1, 6) for v in itertools.product(grid, repeat=k)
    ]
    exact = mixture(rng, 2000)
    exact[rng.choice(2000, 200, replace=False)] = rng.choice([0.0, 1.0], 200)
    columns = [blank(rng, mixture(rng, 2000), f) for f in (0.0, 0.01, 0.1, 0.5)]
    random = []
    for _ in range(50):
        p = mixture(rng, int(rng.integers(2, 2001)))
        if rng.uniform() < 0.5:
            p = np.round(p, int(rng.integers(1, 4)))
        random.append(blank(rng, p, rng.choice([0.0, 0.1])))
    large = mixture(rng, 100_000)
    large[rng.choice(100_000, 10_000, replace=False)] = rng.choice(large, 10_000)
    mixed = np.concatenate([np.round(mixture(rng, 150), 2), mixture(rng, 150)])
    mixed[rng.choice(300, 20, replace=False)] = rng.choice([0.0, 1.0], 20)
    subset = np.round(mixture(rng, 40), 3)
    subset[[3, 17]] = np.nan
    return [
        Case("uniform", "uniform, 2,000 values", [rng.uniform(size=2000)]),
        Case(
            "ties",
            "rounded to 2 decimals (ties, 0, 1), 2,000 values",
            [np.round(mixture(rng, 2000), 2)],
        ),
        Case("missing", "10% NaN, 2,000 values", [blank(rng, mixture(rng, 2000), 0.1)]),
        Case("exact", "10% exactly 0 or 1, 2,000 values", [exact]),
        Case(
            "larger_n",
            "n = 2,000 for 500 values with ties and NaN",
            [blank(rng, np.round(mixture(rng, 500), 3), 0.05)],
            n=2000,
        ),
        Case(
            "small",
            f"all {len(small):,} vectors of length 1 to 5 over "
            "{0, 0.01, 0.03, 0.5, 1, NaN}",
            small,
            layout="rows",
        ),
        Case("small_n", "the same vectors with n = 6", small, n=6, layout="rows"),
        Case(
            "columns",
            "2,000 x 4 matrix by column, 0 to 50% NaN",
            columns,
            layout="columns",
        ),
        Case(
            "random", "50 random families of 2 to 2,000 values", random, layout="rows"
        ),
        Case("random_n", "the same families with n = 2,500", random, 2500, "rows"),
        Case("large", "100,000 values, 10% duplicated", [large]),
        Case(
            "mixed",
            "300 values with ties, 0, 1, NaN",
            [blank(rng, mixed, 0.05)],
            fixture=True,
        ),
        Case(
            "subset",
            "n = 100 for 40 values with ties, NaN",
            [subset],
            n=100,
            fixture=True,
        ),
        Case(
            "columns_small",
            "60 x 3 matrix by column, 0 to 40% NaN",
            [blank(rng, mixture(rng, 60), f) for f in (0.0, 0.1, 0.4)],
            layout="columns",
            fixture=True,
        ),
        Case(
            "small_rows",
            "the vectors of length 1 to 3",
            [v for v in small if v.size <= 3],
            layout="rows",
            fixture=True,
        ),
    ]


def arrange(
    vectors: list[np.ndarray], layout: str, flat: np.ndarray | None = None
) -> np.ndarray:
    """Lay out the vectors, or values stored in their order, as CytePool takes them."""
    flat = np.concatenate(vectors) if flat is None else flat
    parts = np.split(flat, np.cumsum([v.size for v in vectors])[:-1])
    if layout == "vector":
        return parts[0]
    if layout == "columns":
        return np.column_stack(parts)
    rows = np.full((len(parts), max(v.size for v in parts)), np.nan)
    for i, part in enumerate(parts):
        rows[i, : part.size] = part
    return rows


def run_r(cases: list[Case], rscript: str) -> tuple[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        table = ["key n"]
        for key, _, vectors, n, *_ in cases:
            np.concatenate(vectors).astype("<f8").tofile(folder / f"{key}.p")
            (folder / f"{key}.lengths").write_text(
                " ".join(str(v.size) for v in vectors)
            )
            table.append(f"{key} {n}")
        (folder / "cases.tsv").write_text("\n".join(table) + "\n")
        (folder / "p_adjust.R").write_text(R_SCRIPT)
        command = [rscript, "--vanilla", str(folder / "p_adjust.R"), tmp, *METHODS]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            sys.exit(f"R failed:\n{result.stderr}")
        outputs = {
            (key, method): np.fromfile(folder / f"{key}.{method}", "<f8")
            for key, *_ in cases
            for method in METHODS
        }
    return result.stdout.strip(), outputs


def max_relative_difference(ours: np.ndarray, reference: np.ndarray) -> float:
    """Largest |ours - reference| / |reference| (absolute where the reference is 0)."""
    missing = np.isnan(reference)
    if not np.array_equal(np.isnan(ours), missing):
        return np.inf
    a, b = ours[~missing], reference[~missing]
    return float(np.max(np.abs(a - b) / np.where(b == 0, 1.0, np.abs(b)), initial=0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate adjust_p_values against R.")
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    cases = build_cases(np.random.default_rng(20261023))
    version, reference = run_r(cases, os.environ.get("RSCRIPT", "Rscript"))
    print(f"Reference: {version}, stats::p.adjust\n")
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    failed = False
    differing = dict.fromkeys(METHODS, 0)
    compared = 0
    fixtures = {}
    for key, label, vectors, n, layout, fixture in cases:
        p = arrange(vectors, layout)
        worst = 0.0
        for method in METHODS:
            expected = arrange(vectors, layout, reference[key, method])
            start = time.perf_counter()
            ours = adjust_p_values(p, method, n=None if n < 0 else n, axis=AXIS[layout])
            if key == "large" and method == "hommel":
                seconds = time.perf_counter() - start
            worst = max(worst, max_relative_difference(ours, expected))
            differing[method] += np.count_nonzero((ours != expected) & ~np.isnan(p))
            if fixture:
                fixtures[f"{key}/{method}"] = expected
        if fixture:
            fixtures[f"{key}/p"], fixtures[f"{key}/n"] = p, np.array(n)
        compared += np.count_nonzero(~np.isnan(p))
        passed = worst <= TOLERANCE
        failed |= not passed
        print(
            f"| {'fixture: ' * fixture}{label} | R p.adjust | max rel diff, 7 methods "
            f"| {worst:.2g} | {TOLERANCE:g} | {'pass' if passed else 'FAIL'} |"
        )
    counts = ", ".join(f"{method} {count}" for method, count in differing.items())
    print(f"\nValues not bit-identical to R, of {compared:,} per method: {counts}")
    print(f"CytePool Hommel on the 100,000 values: {seconds:.3f} s")
    if args.write_fixtures:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FIXTURES / "p_adjust.npz", **fixtures)
        print(f"Wrote {FIXTURES / 'p_adjust.npz'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
