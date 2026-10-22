"""Check linear model fits and contrasts against limma.

Compares ``cytepool.stats.linear_models.fit_linear_model`` with limma's
``lmFit``, and ``contrasts_fit`` with limma's ``contrasts.fit``, on data
simulated from fixed seeds (2,000 genes, 6 to 20 samples): two groups; several
groups with a covariate; continuous covariates on a large scale; sample
weights; observation weights; about 5% missing values, with genes that lose
rank; zero weights; and a block correlation with and without weights and
missing values.

Every fit is compared on coefficients, ``stdev.unscaled``, ``sigma``,
``df.residual``, ``Amean`` and ``cov.coefficients`` (tier exact, relative
difference at most 1e-10). Coefficients are compared relative to the larger
of their absolute value and their standard error, and covariance matrices
relative to the product of standard deviations. Contrasts are checked in both
modes: ``method="limma"`` against ``contrasts.fit``, and ``method="exact"``
against limma refitted on a reparametrised design whose first coefficient is
the contrast. For genes that lose rank, coefficients that cannot be estimated
are NaN in CytePool; the script confirms that those are exactly the
coefficients whose limma value changes when the columns of the design are
reversed. With ``--benchmark`` it times both packages on 20,000 genes x 200
samples instead.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor package limma. Set ``RSCRIPT`` to the
    Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/linear_models.py
    PYTHONPATH=. python validation/linear_models.py --write-fixtures
    PYTHONPATH=. python validation/linear_models.py --benchmark
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cytepool.stats.linear_models import contrasts_fit, fit_linear_model

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "linear_models"
)
EXACT = 1e-10
N_GENES = 2000
N_FIXTURE_GENES = 50

R_SCRIPT = r"""
suppressPackageStartupMessages(library(limma))
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_doubles <- function(dir, name, n) {
  readBin(file.path(dir, name), what = "double", n = n, size = 8, endian = "little")
}
write_doubles <- function(x, dir, name) {
  writeBin(as.double(x), file.path(dir, name), size = 8, endian = "little")
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  meta <- read.table(file.path(dir, "meta.txt"), header = TRUE, stringsAsFactors = FALSE)
  n <- meta$n_samples; G <- meta$n_genes; p <- meta$n_coefficients; k <- meta$n_contrasts
  y <- matrix(read_doubles(dir, "y.bin", G * n), G, n)
  X <- matrix(read_doubles(dir, "design.bin", n * p), n, p)
  w <- NULL
  if (meta$weights == "sample") w <- read_doubles(dir, "weights.bin", n)
  if (meta$weights == "observation") w <- matrix(read_doubles(dir, "weights.bin", G * n), G, n)
  blk <- NULL
  if (meta$block) blk <- scan(file.path(dir, "block.txt"), quiet = TRUE)
  fit_design <- function(design) {
    if (meta$block) {
      suppressWarnings(lmFit(y, design, weights = w, block = blk, correlation = meta$correlation))
    } else {
      suppressWarnings(lmFit(y, design, weights = w))
    }
  }
  t0 <- proc.time()[["elapsed"]]
  fit <- fit_design(X)
  elapsed <- proc.time()[["elapsed"]] - t0
  write_doubles(fit$coefficients, dir, "limma_coefficients.bin")
  write_doubles(fit$stdev.unscaled, dir, "limma_stdev_unscaled.bin")
  write_doubles(fit$sigma, dir, "limma_sigma.bin")
  write_doubles(fit$df.residual, dir, "limma_df_residual.bin")
  write_doubles(fit$Amean, dir, "limma_amean.bin")
  write_doubles(fit$cov.coefficients, dir, "limma_cov_coefficients.bin")
  write_doubles(elapsed, dir, "limma_seconds.bin")
  if (meta$reversed) {
    reversed <- fit_design(X[, p:1, drop = FALSE])
    write_doubles(reversed$coefficients[, p:1, drop = FALSE], dir, "limma_reversed_coefficients.bin")
  }
  if (k > 0) {
    C <- matrix(read_doubles(dir, "contrasts.bin", p * k), p, k)
    contrasted <- contrasts.fit(fit, C)
    write_doubles(contrasted$coefficients, dir, "limma_contrast_coefficients.bin")
    write_doubles(contrasted$stdev.unscaled, dir, "limma_contrast_stdev_unscaled.bin")
    write_doubles(contrasted$cov.coefficients, dir, "limma_contrast_cov_coefficients.bin")
    exact_coefficients <- matrix(NA_real_, G, k)
    exact_stdev <- matrix(NA_real_, G, k)
    reversed_exact <- matrix(NA_real_, G, k)
    for (j in seq_len(k)) {
      Xj <- matrix(read_doubles(dir, sprintf("reparametrised_%d.bin", j), n * p), n, p)
      refit <- fit_design(Xj)
      exact_coefficients[, j] <- refit$coefficients[, 1]
      exact_stdev[, j] <- refit$stdev.unscaled[, 1]
      if (meta$reversed) reversed_exact[, j] <- fit_design(Xj[, p:1, drop = FALSE])$coefficients[, p]
    }
    write_doubles(exact_coefficients, dir, "limma_refit_coefficients.bin")
    write_doubles(exact_stdev, dir, "limma_refit_stdev_unscaled.bin")
    write_doubles(reversed_exact, dir, "limma_reversed_refit_coefficients.bin")
  }
}
writeLines(
  c(paste("R", as.character(getRversion())), paste("limma", as.character(packageVersion("limma")))),
  file.path(root, "versions.txt")
)
"""


@dataclass
class Case:
    """Inputs of one comparison. Arrays are samples x genes, as in CytePool."""

    name: str
    y: np.ndarray
    design: np.ndarray
    coefficient_names: tuple[str, ...]
    contrasts: np.ndarray
    contrast_names: tuple[str, ...]
    weights: np.ndarray | None = None
    block: np.ndarray | None = None
    correlation: float | None = None
    loses_rank: bool = False
    approximate_contrasts: bool = False


def treatment_design(groups: np.ndarray, *covariates: np.ndarray) -> np.ndarray:
    """Intercept, one indicator per non-reference group, then the covariates."""
    levels = np.unique(groups)
    indicators = [(groups == level).astype(float) for level in levels[1:]]
    return np.column_stack([np.ones(groups.size), *indicators, *covariates])


def simulate(
    rng: np.random.Generator,
    design: np.ndarray,
    n_genes: int,
    effect_scales: list[float],
) -> np.ndarray:
    """Log-expression with gene-specific means, variances and sparse effects."""
    n_samples, n_coefficients = design.shape
    effects = np.zeros((n_coefficients, n_genes))
    effects[0] = rng.uniform(2.0, 12.0, n_genes)
    for j, scale in enumerate(effect_scales, start=1):
        active = rng.random(n_genes) < 0.3
        effects[j] = np.where(active, rng.normal(0.0, scale, n_genes), 0.0)
    sd = 0.3 * np.sqrt(4.0 / rng.chisquare(4.0, n_genes))
    return design @ effects + rng.normal(0.0, 1.0, (n_samples, n_genes)) * sd


def observation_weights(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """Positive weights that rise with expression, like voom's."""
    level = np.nan_to_num(y, nan=6.0)
    return np.exp(rng.normal(0.0, 0.5, y.shape)) / (
        0.2 + 4.0 / (1.0 + np.exp(level - 5.0))
    )


def three_groups(n_per_group: int) -> np.ndarray:
    return np.repeat(np.array(["A", "B", "C"]), n_per_group)


def make_cases(n_genes: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases = []

    groups = np.repeat(np.array(["A", "B"]), 3)
    x = treatment_design(groups)
    cases.append(
        Case(
            "two_groups",
            simulate(rng, x, n_genes, [1.0]),
            x,
            ("Intercept", "groupB"),
            np.array([[0.0, 1.0], [1.0, 1.0]]),
            ("BvsA", "meanB"),
        )
    )

    groups = three_groups(4)
    age = rng.uniform(25.0, 70.0, groups.size)
    x = treatment_design(groups, age)
    covariate_names = ("Intercept", "groupB", "groupC", "age")
    covariate_contrasts = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 50.0]]
    ).T
    covariate_contrast_names = ("BvsA", "CvsB", "B_at_age_50")
    cases.append(
        Case(
            "groups_and_covariate",
            simulate(rng, x, n_genes, [1.0, 1.0, 0.02]),
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
        )
    )

    groups = np.repeat(np.array(["A", "B"]), 10)
    age_days = rng.uniform(20.0, 80.0, groups.size) * 365.25
    library_size = rng.uniform(1e6, 5e7, groups.size)
    year = rng.integers(2001, 2013, groups.size).astype(float)
    large_x = treatment_design(groups, age_days, library_size, year)
    large_names = ("Intercept", "groupB", "age_days", "library_size", "year")
    large_contrasts = np.array(
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 365.25, 0.0, 0.0]]
    ).T
    large_contrast_names = ("BvsA", "age_per_year")
    large_effects = [1.0, 5e-5, 2e-8, 0.05]
    cases.append(
        Case(
            "large_scale_covariates",
            simulate(rng, large_x, n_genes, large_effects),
            large_x,
            large_names,
            large_contrasts,
            large_contrast_names,
        )
    )

    groups = three_groups(4)
    age = rng.uniform(25.0, 70.0, groups.size)
    x = treatment_design(groups, age)
    y = simulate(rng, x, n_genes, [1.0, 1.0, 0.02])
    cases.append(
        Case(
            "sample_weights",
            y,
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
            weights=rng.uniform(0.3, 3.0, groups.size),
        )
    )

    y = simulate(rng, x, n_genes, [1.0, 1.0, 0.02])
    cases.append(
        Case(
            "observation_weights",
            y,
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
            weights=observation_weights(rng, y),
            approximate_contrasts=True,
        )
    )

    means = np.column_stack([(groups == g).astype(float) for g in ("A", "B", "C")])
    y = simulate(rng, means, n_genes, [1.0, 1.0])
    cases.append(
        Case(
            "group_means_observation_weights",
            y,
            means,
            ("A", "B", "C"),
            np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.5, 0.5]]).T,
            ("BvsA", "CvsB", "mean_BC"),
            weights=observation_weights(rng, y),
        )
    )

    y = with_missing_values(rng, simulate(rng, x, n_genes, [1.0, 1.0, 0.02]), groups)
    cases.append(
        Case(
            "missing_values",
            y,
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
            loses_rank=True,
            approximate_contrasts=True,
        )
    )

    y = with_missing_values(rng, simulate(rng, x, n_genes, [1.0, 1.0, 0.02]), groups)
    cases.append(
        Case(
            "missing_values_sample_weights",
            y,
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
            weights=rng.uniform(0.3, 3.0, groups.size),
            loses_rank=True,
            approximate_contrasts=True,
        )
    )

    donors = np.repeat(np.arange(8), 2)
    treated = np.tile([0.0, 1.0], 8)
    donor_age = rng.uniform(25.0, 70.0, 8)[donors]
    x = np.column_stack([np.ones(16), treated, donor_age])
    paired_names = ("Intercept", "treated", "donor_age")
    paired_contrasts = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 50.0]]).T
    paired_contrast_names = ("treated", "treated_at_age_50")
    for name, weights_kind, missing in (
        ("block_correlation", None, False),
        ("block_correlation_sample_weights", "sample", False),
        ("block_correlation_observation_weights", "observation", False),
        ("block_correlation_missing_values", None, True),
    ):
        y = simulate(rng, x, n_genes, [1.0, 0.02])
        y = y + rng.normal(0.0, 0.4, (8, n_genes))[donors]
        if weights_kind == "sample":
            weights = rng.uniform(0.3, 3.0, 16)
        elif weights_kind == "observation":
            weights = observation_weights(rng, y)
        else:
            weights = None
        if missing:
            y = with_missing_values_in_blocks(rng, y, donors, treated)
        cases.append(
            Case(
                name,
                y,
                x,
                paired_names,
                paired_contrasts,
                paired_contrast_names,
                weights=weights,
                block=donors,
                correlation=0.6,
                loses_rank=missing,
                approximate_contrasts=weights_kind == "observation" or missing,
            )
        )

    # Zero weights leave observations out of the fit but not out of Amean.
    groups = three_groups(4)
    x = treatment_design(groups, rng.uniform(25.0, 70.0, groups.size))
    y = simulate(rng, x, n_genes, [1.0, 1.0, 0.02])
    weights = observation_weights(rng, y)
    weights[rng.random(weights.shape) < 0.05] = 0.0
    weights[groups == "A", 0] = 0.0  # the reference group left out of gene 0
    weights[groups == "C", 1] = 0.0  # group C left out of gene 1
    cases.append(
        Case(
            "zero_observation_weights",
            y,
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
            weights=weights,
            loses_rank=True,
            approximate_contrasts=True,
        )
    )
    weights = rng.uniform(0.3, 3.0, groups.size)
    weights[5] = 0.0
    cases.append(
        Case(
            "zero_sample_weight",
            simulate(rng, x, n_genes, [1.0, 1.0, 0.02]),
            x,
            covariate_names,
            covariate_contrasts,
            covariate_contrast_names,
            weights=weights,
            approximate_contrasts=True,
        )
    )

    # Large-scale covariates through the per-feature fits.
    y = simulate(rng, large_x, n_genes, large_effects)
    weights = observation_weights(rng, y)
    y[rng.random(y.shape) < 0.05] = np.nan
    cases.append(
        Case(
            "large_scale_covariates_weights_missing",
            y,
            large_x,
            large_names,
            large_contrasts,
            large_contrast_names,
            weights=weights,
            loses_rank=True,
            approximate_contrasts=True,
        )
    )
    return cases


def with_missing_values(
    rng: np.random.Generator, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """About 5% missing values, plus genes that lose rank or degrees of freedom.

    Genes 0 to 5 get only the missing values planted here.
    """
    y = y.copy()
    random_missing = rng.random(y.shape) < 0.05
    random_missing[:, :6] = False
    y[random_missing] = np.nan
    first = {g: np.flatnonzero(groups == g) for g in ("A", "B", "C")}
    y[first["A"], 0] = np.nan  # reference group missing: nothing is estimable
    y[first["C"], 1] = np.nan  # group C missing: only its coefficient is lost
    keep = [first["A"][0], first["B"][0], first["C"][0]]
    y[np.setdiff1d(np.arange(groups.size), keep), 2] = np.nan  # one sample per group
    keep = [first["A"][0], first["A"][1], first["B"][0], first["C"][0]]
    y[np.setdiff1d(np.arange(groups.size), keep), 3] = np.nan  # exactly identified
    y[:, 4] = np.nan  # nothing observed
    y[1:, 5] = np.nan  # a single observation
    return y


def with_missing_values_in_blocks(
    rng: np.random.Generator, y: np.ndarray, donors: np.ndarray, treated: np.ndarray
) -> np.ndarray:
    """About 5% missing values, plus incomplete and empty blocks.

    Genes 0 to 2 get only the missing values planted here.
    """
    y = y.copy()
    random_missing = rng.random(y.shape) < 0.05
    random_missing[:, :3] = False
    y[random_missing] = np.nan
    y[donors == 0, 0] = np.nan  # a whole block missing
    y[(donors == 1) & (treated == 1.0), 1] = np.nan  # a block reduced to one sample
    y[treated == 1.0, 2] = np.nan  # no treated sample: treated cannot be estimated
    return y


def reparametrised_design(design: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """A design whose first coefficient is the contrast of the original ones."""
    pivot = int(np.argmax(np.abs(contrast)))
    others = [np.eye(contrast.size)[j] for j in range(contrast.size) if j != pivot]
    transform = np.vstack([contrast, *others])
    return design @ np.linalg.inv(transform)


def write_case(case: Case, directory: Path) -> None:
    directory.mkdir(parents=True)
    n_samples, n_genes = case.y.shape
    weights_kind = "none"
    # R reads matrices column by column, with genes as rows.
    if case.weights is not None:
        weights_kind = "sample" if case.weights.ndim == 1 else "observation"
        (directory / "weights.bin").write_bytes(case.weights.T.tobytes(order="F"))
    (directory / "y.bin").write_bytes(case.y.T.tobytes(order="F"))
    (directory / "design.bin").write_bytes(case.design.tobytes(order="F"))
    (directory / "contrasts.bin").write_bytes(case.contrasts.tobytes(order="F"))
    for j in range(case.contrasts.shape[1]):
        design = reparametrised_design(case.design, case.contrasts[:, j])
        (directory / f"reparametrised_{j + 1}.bin").write_bytes(
            design.tobytes(order="F")
        )
    if case.block is not None:
        (directory / "block.txt").write_text(" ".join(str(int(b)) for b in case.block))
    header = "n_samples n_genes n_coefficients n_contrasts weights block correlation reversed"
    row = (
        f"{n_samples} {n_genes} {case.design.shape[1]} {case.contrasts.shape[1]} "
        f"{weights_kind} {'TRUE' if case.block is not None else 'FALSE'} "
        f"{case.correlation if case.correlation is not None else 0.0!r} "
        f"{'TRUE' if case.loses_rank else 'FALSE'}"
    )
    (directory / "meta.txt").write_text(f"{header}\n{row}\n")


def read_doubles(path: Path) -> np.ndarray:
    """Doubles written by R, with its NA (a signalling NaN) made a quiet NaN."""
    values = np.fromfile(path, dtype="<f8")
    values[np.isnan(values)] = np.nan
    return values


def read_matrix(path: Path, n_rows: int, n_columns: int) -> np.ndarray:
    """An R matrix written column by column, as a (rows, columns) array."""
    return read_doubles(path).reshape(n_columns, n_rows).T


def read_reference(case: Case, directory: Path) -> dict[str, np.ndarray]:
    n_samples, n_genes = case.y.shape
    p = case.design.shape[1]
    k = case.contrasts.shape[1]
    reference = {
        "coefficients": read_matrix(directory / "limma_coefficients.bin", n_genes, p),
        "stdev_unscaled": read_matrix(
            directory / "limma_stdev_unscaled.bin", n_genes, p
        ),
        "sigma": read_doubles(directory / "limma_sigma.bin"),
        "df_residual": read_doubles(directory / "limma_df_residual.bin"),
        "mean_expression": read_doubles(directory / "limma_amean.bin"),
        "cov_coefficients": read_matrix(directory / "limma_cov_coefficients.bin", p, p),
        "seconds": read_doubles(directory / "limma_seconds.bin"),
        "contrast_coefficients": read_matrix(
            directory / "limma_contrast_coefficients.bin", n_genes, k
        ),
        "contrast_stdev_unscaled": read_matrix(
            directory / "limma_contrast_stdev_unscaled.bin", n_genes, k
        ),
        "contrast_cov_coefficients": read_matrix(
            directory / "limma_contrast_cov_coefficients.bin", k, k
        ),
        "refit_coefficients": read_matrix(
            directory / "limma_refit_coefficients.bin", n_genes, k
        ),
        "refit_stdev_unscaled": read_matrix(
            directory / "limma_refit_stdev_unscaled.bin", n_genes, k
        ),
    }
    if case.loses_rank:
        reference["reversed_coefficients"] = read_matrix(
            directory / "limma_reversed_coefficients.bin", n_genes, p
        )
        reference["reversed_refit_coefficients"] = read_matrix(
            directory / "limma_reversed_refit_coefficients.bin", n_genes, k
        )
    return reference


def run_limma(cases: list[Case]) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    rscript = os.environ.get("RSCRIPT", "Rscript")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for case in cases:
            write_case(case, root / case.name)
        (root / "cases.txt").write_text("\n".join(case.name for case in cases) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        references = [read_reference(case, root / case.name) for case in cases]
        versions = (root / "versions.txt").read_text().split("\n")
    return references, [v for v in versions if v]


def fit_case(case: Case):
    return fit_linear_model(
        case.y,
        case.design,
        weights=case.weights,
        block=case.block,
        correlation=case.correlation,
        coefficient_names=case.coefficient_names,
    )


def relative_difference(
    actual: np.ndarray, desired: np.ndarray, expected_nan: np.ndarray | None = None
) -> float:
    """Largest elementwise relative difference.

    Infinite when NaN marks other entries than ``expected_nan`` (by default the
    NaN entries of ``desired``), or when ``desired`` is NaN where ``actual`` is
    not.
    """
    return _scaled_difference(actual, desired, np.abs(desired), expected_nan)


def coefficient_difference(
    actual: np.ndarray,
    desired: np.ndarray,
    standard_error: np.ndarray,
    expected_nan: np.ndarray | None = None,
) -> float:
    """Largest difference relative to max(|desired|, standard error).

    A coefficient much smaller than its standard error carries no relative
    precision: rounding moves it, in any implementation, by a small multiple
    of machine precision times its standard error. Larger coefficients are
    compared by their relative difference.
    """
    return _scaled_difference(
        actual, desired, np.fmax(np.abs(desired), standard_error), expected_nan
    )


def _scaled_difference(
    actual: np.ndarray,
    desired: np.ndarray,
    scale: np.ndarray,
    expected_nan: np.ndarray | None,
) -> float:
    if expected_nan is None:
        expected_nan = np.isnan(desired)
    if not np.array_equal(np.isnan(actual), expected_nan):
        return np.inf
    finite = ~expected_nan
    if np.isnan(desired[finite]).any():
        return np.inf
    if not finite.any():
        return 0.0
    difference = np.abs(actual[finite] - desired[finite])
    scale = np.where(scale[finite] > 0.0, scale[finite], 1.0)
    return float(np.max(difference / scale))


def covariance_difference(actual: np.ndarray, desired: np.ndarray) -> float:
    """Largest difference relative to the product of standard deviations."""
    scale = np.sqrt(np.outer(np.diag(desired), np.diag(desired)))
    return float(np.max(np.abs(actual - desired) / scale))


class Table:
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

    def count(self, check: str, reference: str, metric: str, value: int) -> None:
        passed = value == 0
        self.failures += not passed
        self.rows.append(
            (check, reference, metric, str(value), "0", "pass" if passed else "FAIL")
        )

    def info(self, check: str, reference: str, metric: str, value: str) -> None:
        self.rows.append((check, reference, metric, value, "n/a", "info"))

    def print(self) -> None:
        print("| check | reference | metric | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


RELATIVE = "max rel. difference"
COEFFICIENT = "max diff. / max(abs, se)"
COVARIANCE = "max diff. / (sd x sd)"


def not_estimable(values: np.ndarray, reversed_values: np.ndarray) -> np.ndarray:
    """Entries that limma leaves NA, or whose value depends on the column order.

    An estimable coefficient or contrast has one least squares estimate, so
    reversing the columns of the design leaves limma's value unchanged; for
    the others limma reports an arbitrary solution, or NA.
    """
    with np.errstate(invalid="ignore"):
        spread = np.abs(values - reversed_values)
    order_dependent = np.isnan(values) | np.isnan(reversed_values)
    finite = ~order_dependent
    scale = np.maximum(np.abs(values[finite]), 1.0)
    order_dependent[finite] = spread[finite] > 1e-8 * scale
    return order_dependent


def compare(case: Case, reference: dict[str, np.ndarray], table: Table) -> None:
    with np.errstate(invalid="ignore", divide="ignore"):
        _compare(case, reference, table)


def _compare(case: Case, reference: dict[str, np.ndarray], table: Table) -> None:
    fit = fit_case(case)
    name = case.name
    lmfit = "lmFit"
    limma_nan = np.isnan(reference["coefficients"])
    if case.loses_rank:
        expected_nan = not_estimable(
            reference["coefficients"], reference["reversed_coefficients"]
        )
        coefficient_check = f"{name}: coefficients; NaN exactly where not estimable"
        reference_name = "lmFit, also with columns reversed"
    else:
        expected_nan = limma_nan
        coefficient_check = f"{name}: coefficients"
        reference_name = lmfit
    standard_error = reference["stdev_unscaled"] * reference["sigma"][:, np.newaxis]
    table.check(
        coefficient_check,
        reference_name,
        COEFFICIENT,
        coefficient_difference(
            fit.coefficients, reference["coefficients"], standard_error, expected_nan
        ),
        EXACT,
    )
    if case.loses_rank:
        table.info(
            f"{name}: non-estimable coefficients that limma reports",
            lmfit,
            "entries",
            str(int(np.count_nonzero(expected_nan & ~limma_nan))),
        )
    table.check(
        f"{name}: stdev_unscaled",
        lmfit,
        RELATIVE,
        relative_difference(
            fit.stdev_unscaled, reference["stdev_unscaled"], expected_nan
        ),
        EXACT,
    )
    table.check(
        f"{name}: sigma",
        lmfit,
        RELATIVE,
        relative_difference(fit.sigma, reference["sigma"]),
        EXACT,
    )
    table.count(
        f"{name}: df_residual",
        lmfit,
        "genes that differ",
        int(np.count_nonzero(fit.df_residual != reference["df_residual"])),
    )
    table.check(
        f"{name}: mean_expression (Amean)",
        lmfit,
        RELATIVE,
        relative_difference(fit.mean_expression, reference["mean_expression"]),
        EXACT,
    )
    table.check(
        f"{name}: cov_coefficients",
        lmfit,
        COVARIANCE,
        covariance_difference(fit.cov_coefficients, reference["cov_coefficients"]),
        EXACT,
    )

    # A contrast is NaN when it puts weight on a coefficient that is NaN.
    involved = (case.contrasts != 0.0).astype(float)
    expected_contrast_nan = expected_nan.astype(float) @ involved > 0.0
    limma_mode = contrasts_fit(
        fit, case.contrasts, names=case.contrast_names, method="limma"
    )
    contrasts = "contrasts.fit"
    contrast_se = (
        reference["contrast_stdev_unscaled"] * reference["sigma"][:, np.newaxis]
    )
    table.check(
        f"{name}: contrasts, limma method: coefficients",
        contrasts,
        COEFFICIENT,
        coefficient_difference(
            limma_mode.coefficients,
            reference["contrast_coefficients"],
            contrast_se,
            expected_contrast_nan,
        ),
        EXACT,
    )
    table.check(
        f"{name}: contrasts, limma method: stdev_unscaled",
        contrasts,
        RELATIVE,
        relative_difference(
            limma_mode.stdev_unscaled,
            reference["contrast_stdev_unscaled"],
            expected_contrast_nan,
        ),
        EXACT,
    )
    table.check(
        f"{name}: contrasts: cov_coefficients",
        contrasts,
        COVARIANCE,
        covariance_difference(
            limma_mode.cov_coefficients, reference["contrast_cov_coefficients"]
        ),
        EXACT,
    )
    exact_mode = contrasts_fit(
        fit, case.contrasts, names=case.contrast_names, method="exact"
    )
    refit = "lmFit, reparametrised design"
    refit_se = reference["refit_stdev_unscaled"] * reference["sigma"][:, np.newaxis]
    table.check(
        f"{name}: contrasts, exact method: coefficients",
        refit,
        COEFFICIENT,
        coefficient_difference(
            exact_mode.coefficients,
            reference["refit_coefficients"],
            refit_se,
            expected_contrast_nan,
        ),
        EXACT,
    )
    table.check(
        f"{name}: contrasts, exact method: stdev_unscaled",
        refit,
        RELATIVE,
        relative_difference(
            exact_mode.stdev_unscaled,
            reference["refit_stdev_unscaled"],
            expected_contrast_nan,
        ),
        EXACT,
    )
    if case.loses_rank:
        estimable = ~not_estimable(
            reference["refit_coefficients"], reference["reversed_refit_coefficients"]
        )
        reported = ~np.isnan(exact_mode.coefficients)
        table.count(
            f"{name}: contrasts, exact method: finite but not estimable",
            "lmFit, reparametrised, columns reversed",
            "entries",
            int(np.count_nonzero(reported & ~estimable)),
        )
        table.info(
            f"{name}: estimable contrasts of non-estimable coefficients, left NaN",
            "lmFit, reparametrised, columns reversed",
            "entries",
            str(int(np.count_nonzero(~reported & estimable))),
        )
    if case.approximate_contrasts:
        ratio = reference["contrast_stdev_unscaled"] / reference["refit_stdev_unscaled"]
        ratio = ratio[np.isfinite(ratio)]
        off = np.mean(np.abs(ratio - 1.0) > 0.05)
        table.info(
            f"{name}: limma contrast stdev_unscaled / exact",
            "contrasts.fit, refit",
            "range; share off by > 5%",
            f"{ratio.min():.2f} to {ratio.max():.2f}; {off:.0%}",
        )


def write_fixtures(cases: list[Case], references: list[dict[str, np.ndarray]]) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for case, reference in zip(cases, references):
        arrays = {
            "y": case.y,
            "design": case.design,
            "coefficient_names": np.array(case.coefficient_names),
            "contrasts": case.contrasts,
            "contrast_names": np.array(case.contrast_names),
            **{key: value for key, value in reference.items() if key != "seconds"},
        }
        if case.weights is not None:
            arrays["weights"] = case.weights
        if case.block is not None:
            arrays["block"] = case.block
            arrays["correlation"] = np.array(case.correlation)
        np.savez_compressed(FIXTURE_DIRECTORY / f"{case.name}.npz", **arrays)
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIRECTORY}")


def benchmark(n_genes: int = 20_000, n_samples: int = 200) -> None:
    """Time CytePool and limma on a realistic size."""
    rng = np.random.default_rng(20261022)
    groups = np.repeat(np.array(["A", "B", "C", "D"]), n_samples // 4)
    donors = np.repeat(np.arange(n_samples // 2), 2)
    x = treatment_design(
        groups, rng.normal(50.0, 10.0, n_samples), rng.normal(0.0, 1.0, n_samples)
    )
    y = simulate(rng, x, n_genes, [1.0, 1.0, 1.0, 0.02, 0.2])
    weights = observation_weights(rng, y)
    missing = y.copy()
    missing[rng.random(y.shape) < 0.05] = np.nan
    names = tuple(f"x{j + 1}" for j in range(x.shape[1]))
    contrast = np.zeros((x.shape[1], 1))
    contrast[1, 0] = 1.0
    scenarios = [
        Case("ordinary least squares", y, x, names, contrast, ("c",)),
        Case("observation weights", y, x, names, contrast, ("c",), weights=weights),
        Case("5% missing values", missing, x, names, contrast, ("c",)),
        Case(
            "block correlation",
            y,
            x,
            names,
            contrast,
            ("c",),
            block=donors,
            correlation=0.3,
        ),
        Case(
            "block correlation, observation weights",
            y,
            x,
            names,
            contrast,
            ("c",),
            weights=weights,
            block=donors,
            correlation=0.3,
        ),
    ]
    for scenario in scenarios:
        scenario.name = (
            scenario.name.replace(" ", "_").replace(",", "").replace("%", "pct")
        )
    references, versions = run_limma(scenarios)
    print(f"Benchmark: {n_genes} genes x {n_samples} samples; {', '.join(versions)}")
    print("| scenario | limma lmFit (s) | CytePool (s) | speed-up |")
    print("| --- | --- | --- | --- |")
    for scenario, reference in zip(scenarios, references):
        fit_case(scenario)  # compile and warm up
        seconds = []
        for _ in range(3):
            start = time.perf_counter()
            fit_case(scenario)
            seconds.append(time.perf_counter() - start)
        ours = float(np.median(seconds))
        theirs = float(reference["seconds"][0])
        print(f"| {scenario.name} | {theirs:.2f} | {ours:.3f} | {theirs / ours:.0f}x |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="time 20,000 genes x 200 samples"
    )
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark()
        return 0
    cases = make_cases(N_GENES, seed=1)
    fixture_cases = (
        make_cases(N_FIXTURE_GENES, seed=2) if arguments.write_fixtures else []
    )
    for case in fixture_cases:
        case.name = f"fixture_{case.name}"
    references, versions = run_limma(cases + fixture_cases)
    print("Reference: " + ", ".join(versions))
    table = Table()
    for case, reference in zip(cases + fixture_cases, references):
        compare(case, reference, table)
    table.print()
    if arguments.write_fixtures:
        if table.failures:
            print("not writing fixtures: some checks failed")
        else:
            for case in fixture_cases:
                case.name = case.name.removeprefix("fixture_")
            write_fixtures(fixture_cases, references[len(cases) :])
    print(f"{len(table.rows)} checks, {table.failures} failed")
    return 1 if table.failures else 0


if __name__ == "__main__":
    sys.exit(main())
