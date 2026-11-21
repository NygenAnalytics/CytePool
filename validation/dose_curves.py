"""Check fit_dose_curves and the curves of dose_response against pybmds, drc, DRomics,
scipy and limma.

Five dose series simulated by ``simulate_dose_series`` from fixed seeds (20261220 to
20261224): ``ladder``, the vehicle and 8 half-log doses from 0.01 to 30 ug/kg x 3, 300
genes, a cell type of every sample and one that loses the top dose; ``decades``, the
vehicle and 4 decades x 3; ``four``, the vehicle and 3 decades x 2; ``descriptive``,
the vehicle x 3 and 5 doses x 1, a cell type that supports only description; and
``sex``, ``"~ sex"`` with 2 females and 2 males at each of 5 doses.

Under constant variance, each gene's values in the cell type's fit by limma_voom (36
genes of the first three series, a spread of the true shapes and flat genes) are fitted
by CytePool and, as separate programs seeing only the doses and values, by pybmds 25.2
(``"epa"``'s linear, polynomial 2, power, Hill and exponential 3 and 5, normal errors of
constant variance, default bounds, every parameter counted in the AIC, each model run
increasing and decreasing and the larger maximum kept), drc 4.0-0's ``LL.4`` with
``drmc(relTol = 1e-12, maxIt = 10000)``, and DRomics 2.6.3's ``drcfit`` (defaults,
``postfitfilter = FALSE`` and a very low ``deltaAICminfromnullmodel``, so that every
item's criteria are reported). In the weighted and moderated modes, dose_response's
dose-group means, covariance and variances of the ladder's two cell types and of the
series with sex (each mode and set, a spread of 6 genes each) are compared with limma's
(voom, lmFit with the dose groups' indicators, each gene's lm.wfit covariance,
eBayes), and every set in every mode with scipy's least_squares from 200 random starts
within the documented bounds on the same means and covariance, run in another process
of this script that never imports CytePool. The means and bounds of every model are
written here from the documentation, apart from CytePool's.

Tolerances: a fit's maximum may not fall below a reference's by more than 1e-8 of the
residual variance under constant variance, or of Q / (K s^2) in the other modes, which
are scale-free where a relative difference of log-likelihoods near 0 is not; a lower
reference maximum prints as a deliberate difference. Parameters of fits that reach the
same maximum off every bound are compared in standard errors: a log-likelihood within
dLL of its maximum puts a parameter within sqrt(2 dLL) standard errors of it, so the
rows bound |d theta| / (SE sqrt(2 max(dLL, 1e-12))) by 10, with pybmds's and drc's own
standard errors and, for DRomics, which reports none, CytePool's from the documented
formula's Jacobian; DRomics prints its criteria to two decimals, which sets its dLL at
0.0025. drc's LL.4 letters (b, c, d, e) with b < 0 are CytePool's (-b, d, c, e). DRomics' criteria may lie below CytePool's AICc by their rounding, 0.005, and then by
1e-8 per sample. The quantities that are closed forms of the fit (AIC's count, test
4's degrees of freedom, limma's means, covariances and F-tests, and the F-test of lack
of fit under constant variance against R's ``anova`` of nested linear models) are exact
(1e-10) or identical; pybmds's test 4, the likelihood ratio n log(1 + (K - p) F / (n -
K)) of CytePool's F on a chi-square, and scaled residuals are close (1e-6), as the
maxima agree to 1e-8; the dose of a turn against DRomics' ``xextrem`` is close within
1e-3, the convergence of its ``nls`` fits. Whether each best fit turns, strictly
between 0 and the top dose, and which dose group drives a turn are identical to the
rules coded here: the lowest dose whose removal leaves a best fit of the whole
"dromics" set by AICc, refitted by scipy from 100 starts on the other K - 1 groups,
that does not turn within the doses kept, and none when no model that can turn has a
spare group among K - 1, as at 5 dose groups. Besides the series' genes, genes that
turn, or do not, by construction (a raised dose group, bells over three groups, rises
at the top dose alone and flat genes) are fitted at 5, 6 and 9 dose groups under
constant variance, and at 9 and 5 in the weighted mode with the covariance of the
ladder's and the sex series' fits.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root, with scipy 1.18.1
    for the multi-start reference.
    pybmds 25.2 (``BMDS_PYTHON``, a Python with pybmds; default ``python``).
    R 4.4 with drc 4.0-0, DRomics 2.6.3 and jsonlite (``DOSE_RSCRIPT``), and limma,
    edgeR and jsonlite (``RSCRIPT``); each variable names an Rscript executable and
    defaults to ``Rscript``.

Usage:
    PYTHONPATH=. python validation/dose_curves.py
    PYTHONPATH=. python validation/dose_curves.py --write-fixtures
    PYTHONPATH=. python validation/dose_curves.py --benchmark
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_curves"
)
STARTS = 200
TURN_STARTS = 100
PLUGIN_GENES = 4

# ---------------------------------------------------------------------------------------
# The references, which never import CytePool
# ---------------------------------------------------------------------------------------


def documented_mean(model: str, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each model's mean as the chapter documents it, written apart from CytePool."""
    from scipy.special import ndtr

    t = theta
    with np.errstate(all="ignore"):
        if model in ("linear", "dromics_linear"):
            # g + b1 x, and DRomics' d + b x, both with the level first.
            return t[0] + t[1] * x
        if model == "polynomial_2":
            return t[0] + t[1] * x + t[2] * x * x
        if model == "power":
            return t[0] + t[1] * x ** t[2]
        if model == "hill":
            g, v, k, n = t
            ratio = np.where(x > 0, x**n / (k**n + x**n), 0.0)
            ratio = np.where((x > 0) & (k == 0), 1.0, ratio)
            return g + v * ratio
        if model == "exponential_2":
            return t[0] * np.exp(t[1] * x)
        if model == "exponential_3":
            a, b, d = t
            return a * np.exp(np.sign(b) * (abs(b) * x) ** d)
        if model == "exponential_4":
            a, b, c = t
            return a * (c - (c - 1) * np.exp(-b * x))
        if model == "exponential_5":
            a, b, c, d = t
            return a * (c - (c - 1) * np.exp(-((b * x) ** d)))
        if model == "ll4":
            # c + (d - c) / (1 + u) with u = (x/e)^b, written as c u / (1 + u) + d / (1 +
            # u), which keeps its digits when c and d are far apart.
            b, c, d, e = t
            u = (x / e) ** b
            return c * (u / (1 + u)) + d / (1 + u)
        if model == "exponential":
            # Searched as d + beta (exp(s x) - 1) / s, reported as b = beta / s, e = 1/s.
            beta, d, s = t
            return d + beta * np.expm1(s * x) / s
        if "gauss_probit" in model:
            if model.endswith("symmetric"):
                b, d, e, f = t
                c = d
            elif model.endswith("monotone"):
                b, c, d, e = t
                f = 0.0
            else:
                b, c, d, e, f = t
            if model.startswith("log"):
                z = np.where(
                    x > 0, (np.log(np.maximum(x, 1e-300)) - np.log(e)) / b, -np.inf
                )
            else:
                z = (x - e) / b
            return f * np.exp(-z * z / 2) + d + (c - d) * ndtr(z)
    raise ValueError(model)


def documented_bounds(model: str, doses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each model's bounds as the chapter documents them, from the doses."""
    inf = np.inf
    levels = np.unique(doses)
    top = levels[-1]
    positive = levels[levels > 0]
    lowest = positive[0]
    logs = np.log(positive)
    gap = np.max(np.diff(levels))
    log_gap = np.max(np.diff(logs)) if logs.size > 1 else inf
    rate = 100 / top
    table = {
        "linear": [(-inf, inf)] * 2,
        "dromics_linear": [(-inf, inf)] * 2,
        "polynomial_2": [(-inf, inf)] * 3,
        "power": [(-inf, inf), (-inf, inf), (1, 18)],
        "hill": [(-inf, inf), (-inf, inf), (0, 5 * top), (1, 18)],
        "exponential_2": [(0, inf), (-rate, rate)],
        "exponential_3": [(0, inf), (-rate, rate), (1, 18)],
        "exponential_4": [(0, inf), (0, rate), (0, 1000)],
        "exponential_5": [(0, inf), (0, rate), (0, 1000), (1, 18)],
        "ll4": [(0.1, 18), (-inf, inf), (-inf, inf), (lowest / 100, 5 * top)],
        "exponential": [(-inf, inf), (-inf, inf), (-rate, rate)],
    }
    if model in table:
        pairs = table[model]
    else:
        log = model.startswith("log")
        width = (log_gap / 3, 10 * (logs[-1] - logs[0])) if log else (gap / 3, 10 * top)
        peak = (lowest, top) if log else (0.0, top)
        free = (-inf, inf)
        if model.endswith("symmetric"):
            pairs = [width, free, peak, free]
        elif model.endswith("monotone"):
            pairs = [width, free, free, peak]
        else:
            pairs = [width, free, free, peak, free]
    return np.array([p[0] for p in pairs], float), np.array(
        [p[1] for p in pairs], float
    )


def _start(
    lower: np.ndarray, upper: np.ndarray, level: float, span: float, rng: Any
) -> np.ndarray:
    """A random start within the bounds: log-uniform for positive bounded parameters,
    uniform for others bounded, and around the data's level otherwise."""
    x0 = np.empty(lower.size)
    for j, (a, b) in enumerate(zip(lower, upper)):
        if np.isfinite(a) and np.isfinite(b):
            if a > 0:
                x0[j] = math.exp(rng.uniform(math.log(a), math.log(b)))
            elif b < 0:
                x0[j] = -math.exp(rng.uniform(math.log(-b), math.log(-a)))
            else:
                x0[j] = rng.uniform(a, b)
        elif np.isfinite(a):
            x0[j] = a + abs(level) * rng.uniform(0.5, 1.5) + 1e-3
        else:
            x0[j] = level + span * rng.normal()
    inner_lower = np.where(np.isfinite(lower), lower + 1e-9 * (1 + abs(lower)), -np.inf)
    inner_upper = np.where(np.isfinite(upper), upper - 1e-9 * (1 + abs(upper)), np.inf)
    return np.clip(x0, inner_lower, inner_upper)


def multistart(task: dict[str, Any]) -> dict[str, Any]:
    """The smallest Q = ||R (ybar - f)||^2 of a model from ``starts`` random starts
    within its bounds, by scipy's least_squares (trf, tolerances 1e-15)."""
    from scipy.optimize import least_squares

    model = task["model"]
    doses = np.asarray(task["doses"], float)
    means = np.asarray(task["means"], float)
    root = np.asarray(task["root"], float)
    lower, upper = documented_bounds(model, doses)
    boxes = [(lower, upper)]
    if model == "exponential":
        # The rate keeps its sign, at least 1e-4 / D in size.
        gap = 1e-4 / doses.max()
        positive, negative = (lower.copy(), upper.copy()), (lower.copy(), upper.copy())
        positive[0][2], negative[1][2] = gap, -gap
        boxes = [positive, negative]
    elif model == "polynomial_2":
        # b1 and b2 share a sign: both at least 0, or both at most 0.
        positive, negative = (lower.copy(), upper.copy()), (lower.copy(), upper.copy())
        positive[0][1:] = 0.0
        negative[1][1:] = 0.0
        boxes = [positive, negative]
    rng = np.random.default_rng(task["seed"])
    level, span = float(means.mean()), float(np.ptp(means) + 0.1)
    best_value, best_theta = math.inf, None
    for lo, hi in boxes:
        for _ in range(task["starts"] // len(boxes)):
            x0 = _start(lo, hi, level, span, rng)

            def residuals(theta: np.ndarray) -> np.ndarray:
                r = root @ (means - documented_mean(model, theta, doses))
                return np.where(np.isfinite(r), r, 1e150)

            try:
                fit = least_squares(
                    residuals,
                    x0,
                    bounds=(lo, hi),
                    method="trf",
                    xtol=1e-15,
                    ftol=1e-15,
                    gtol=1e-15,
                    max_nfev=2000,
                )
            except ValueError:
                continue
            value = float(fit.fun @ fit.fun)
            if np.isfinite(value) and value < best_value:
                best_value, best_theta = value, fit.x.tolist()
    return {"key": task["key"], "q": best_value, "theta": best_theta}


def scipy_main(directory: Path) -> None:
    """The multi-start optimum of every task, in parallel processes."""
    from concurrent.futures import ProcessPoolExecutor
    import scipy

    tasks = json.loads((directory / "scipy_inputs.json").read_text())
    with ProcessPoolExecutor(
        max_workers=int(os.environ.get("SCIPY_WORKERS", "4"))
    ) as pool:
        results = list(pool.map(multistart, tasks, chunksize=4))
    out = {"versions": {"scipy": scipy.__version__}, "results": results}
    (directory / "scipy.json").write_text(json.dumps(out))


def _pybmds_fit(task: dict[str, Any]) -> dict[str, Any]:
    """One pybmds fit of a gene's samples, normal errors of constant variance, the
    default bounds, every parameter counted in the AIC."""
    import pybmds
    from pybmds.models import continuous
    from pybmds.models.continuous import DistType

    classes = {
        "linear": continuous.Linear,
        "polynomial_2": continuous.Polynomial,
        "power": continuous.Power,
        "hill": continuous.Hill,
        "exponential_3": continuous.ExponentialM3,
        "exponential_5": continuous.ExponentialM5,
    }
    dataset = pybmds.ContinuousIndividualDataset(
        doses=task["doses"], responses=task["values"]
    )
    settings = {
        "disttype": DistType.normal,
        "count_all_parameters_on_boundary": True,
        "is_increasing": task["increasing"],
    }
    if task["model"] == "polynomial_2":
        settings["degree"] = 2
    model = classes[task["model"]](dataset, settings=settings)
    model.execute()
    result = model.results.model_dump()
    parameters = result["parameters"]
    return {
        "key": task["key"],
        "log_likelihood": result["fit"]["loglikelihood"],
        "aic": result["fit"]["aic"],
        "names": parameters["names"],
        "values": parameters["values"],
        "se": parameters["se"],
        "bounded": parameters["bounded"],
        "test4_p": result["tests"]["p_values"][3],
        "test4_df": result["tests"]["dfs"][3],
        "scaled_residuals": result["gof"]["residual"],
    }


def pybmds_main(directory: Path) -> None:
    """pybmds's fits of every task, in parallel processes."""
    from concurrent.futures import ProcessPoolExecutor
    from importlib.metadata import version

    tasks = json.loads((directory / "pybmds_inputs.json").read_text())
    with ProcessPoolExecutor(
        max_workers=int(os.environ.get("SCIPY_WORKERS", "4"))
    ) as pool:
        results = list(pool.map(_pybmds_fit, tasks, chunksize=2))
    versions = {"pybmds": version("pybmds"), "python": sys.version.split()[0]}
    (directory / "pybmds.json").write_text(
        json.dumps({"versions": versions, "results": results})
    )


DOSE_R = r"""
suppressPackageStartupMessages({library(jsonlite); library(drc); library(DRomics)})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
out <- list(versions = list(R = as.character(getRversion()),
  drc = as.character(packageVersion("drc")), DRomics = as.character(packageVersion("DRomics"))),
  series = list())
for (name in names(input$series)) {
  s <- input$series[[name]]
  # read_json gives an array of equal arrays as a matrix, one row per inner array.
  doses <- as.numeric(s$doses); values <- as.matrix(s$values)
  stopifnot(ncol(values) == length(doses))
  drc_rows <- list()
  for (i in seq_len(nrow(values))) {
    y <- values[i, ]
    fit <- tryCatch(drm(y ~ doses, fct = LL.4(), control = drmc(relTol = 1e-12, maxIt = 10000,
      errorm = FALSE, noMessage = TRUE)), error = function(e) conditionMessage(e))
    if (is.character(fit) || !inherits(fit, "drc")) {
      drc_rows[[i]] <- list(failed = TRUE)
    } else {
      se <- tryCatch(sqrt(diag(vcov(fit))), error = function(e) rep(NA_real_, 4))
      drc_rows[[i]] <- list(failed = FALSE, coef = unname(coef(fit)), se = unname(se),
        loglik = as.numeric(logLik(fit)))
    }
  }
  # The pure-error F of lack of fit of a line and a quadratic: each against free means
  # of the dose groups.
  groups <- factor(doses)
  lof_linear <- numeric(nrow(values)); lof_quadratic <- numeric(nrow(values))
  for (i in seq_len(nrow(values))) {
    y <- values[i, ]
    free <- lm(y ~ groups)
    lof_linear[i] <- anova(lm(y ~ doses), free)[2, "Pr(>F)"]
    lof_quadratic[i] <- anova(lm(y ~ doses + I(doses^2)), free)[2, "Pr(>F)"]
  }
  frame <- data.frame(item = c("dose", paste0("g", seq_len(nrow(values)) - 1)),
                      rbind(doses, values), check.names = FALSE)
  o <- continuousomicdata(frame, check = TRUE)
  selected <- itemselect(o, select.method = "quadratic", FDR = 0.99999999)
  fits <- drcfit(selected, progressbar = FALSE, postfitfilter = FALSE,
                 deltaAICminfromnullmodel = -1e9)
  ic <- as.data.frame(fits$information.criterion.val)
  message("drcfit returns: ", paste(names(fits), collapse = ", "), "; criteria rows ",
          nrow(ic), ", items selected ", length(selected$selectindex))
  ic$id <- as.character(selected$omicdata$item[selected$selectindex])
  out$series[[name]] <- list(drc = drc_rows, fitres = fits$fitres, criteria = ic,
                             unfitted = fits$unfitres, lof_linear = lof_linear,
                             lof_quadratic = lof_quadratic)
}
write_json(out, args[[2]], digits = NA, auto_unbox = TRUE, null = "null", na = "null")
"""


LIMMA_R = r"""
suppressPackageStartupMessages({library(jsonlite); library(limma); library(edgeR)})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
out <- list(versions = list(R = as.character(getRversion()),
  limma = as.character(packageVersion("limma")), edgeR = as.character(packageVersion("edgeR"))),
  jobs = list())
for (name in names(input$jobs)) {
  job <- input$jobs[[name]]
  # read_json gives an array of equal arrays as a matrix, one row per inner array.
  y <- as.matrix(job$counts); X <- as.matrix(job$design)
  stopifnot(nrow(y) == job$n_samples, nrow(X) == job$n_samples)
  dge <- DGEList(t(y)); keep <- filterByExpr(dge, design = X)
  dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
  res <- list(kept = which(keep) - 1L)
  if (job$what == "values") {
    if (job$analysis == "voom") {
      res$values <- voom(dge, X, adaptive.span = TRUE)$E
    } else {
      res$values <- cpm(dge, log = TRUE, prior.count = 3)
    }
    out$jobs[[name]] <- res
    next
  }
  L <- as.matrix(job$contrasts); lof <- as.matrix(job$lack_of_fit)
  stopifnot(nrow(L) == ncol(X), nrow(lof) == ncol(L))
  if (job$analysis == "voom") {
    v <- voom(dge, X, adaptive.span = TRUE); E <- v$E; W <- v$weights; fit <- lmFit(v, X)
  } else {
    E <- cpm(dge, log = TRUE, prior.count = 3); W <- NULL; fit <- lmFit(E, X)
  }
  cfit <- contrasts.fit(fit, L)
  eb <- eBayes(cfit, trend = job$analysis != "voom")
  ngenes <- nrow(E)
  theta <- matrix(0, ngenes, ncol(L)); cov <- vector("list", ngenes)
  for (g in seq_len(ngenes)) {
    f <- if (is.null(W)) lm.fit(X, E[g, ]) else lm.wfit(X, E[g, ], W[g, ])
    stopifnot(identical(f$qr$pivot, seq_len(ncol(X))))
    V <- chol2inv(qr.R(f$qr))
    theta[g, ] <- crossprod(L, f$coefficients)
    cov[[g]] <- t(L) %*% V %*% L
  }
  # The lack-of-fit F of a straight line through the dose-group means: the contrasts
  # lof span the null space of [1, x] at the doses, on the means ybar = M theta.
  M <- diag(ncol(L)); M[, 1] <- 1
  K <- ncol(L); df1 <- ncol(lof)
  lof_f <- function(s2) sapply(seq_len(ngenes), function(g) {
    est <- crossprod(lof, M %*% theta[g, ])
    C <- M %*% cov[[g]] %*% t(M)
    drop(crossprod(est, solve(t(lof) %*% C %*% lof, est))) / df1 / s2[g]
  })
  res$theta <- theta; res$covariance <- lapply(cov, function(m) unname(m))
  res$sigma2 <- fit$sigma^2; res$df_residual <- fit$df.residual
  res$s2_post <- eb$s2.post; res$df_prior <- eb$df.prior
  res$lof_f_weighted <- lof_f(fit$sigma^2); res$lof_f_moderated <- lof_f(eb$s2.post)
  if (job$analysis != "voom") {
    # limma's own F-test of the same contrasts, exact without voom's weights.
    lof_coef <- L %*% t(M) %*% lof
    eb2 <- eBayes(contrasts.fit(fit, lof_coef), trend = TRUE)
    res$lof_f_limma <- as.numeric(classifyTestsF(eb2, fstat.only = TRUE))
  }
  out$jobs[[name]] <- res
}
write_json(out, args[[2]], digits = NA, auto_unbox = TRUE, null = "null", na = "null")
"""


def run_r(program: str, payload: dict[str, Any], variable: str) -> dict[str, Any]:
    """Run an R program on a JSON payload; the variable names its Rscript."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "program.R").write_text(program)
        (directory / "input.json").write_text(json.dumps(payload))
        command = [
            os.environ.get(variable, "Rscript"),
            str(directory / "program.R"),
            str(directory / "input.json"),
            str(directory / "output.json"),
        ]
        subprocess.run(command, check=True)
        result: dict[str, Any] = json.loads((directory / "output.json").read_text())
    return result


def run_python(
    flag: str, interpreter: str, inputs: dict[str, str], output: str
) -> dict[str, Any]:
    """Run this script's reference part in another interpreter on the inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, text in inputs.items():
            (directory / name).write_text(text)
        subprocess.run([interpreter, "-I", __file__, flag, str(directory)], check=True)
        result: dict[str, Any] = json.loads((directory / output).read_text())
    return result


# ---------------------------------------------------------------------------------------
# CytePool
# ---------------------------------------------------------------------------------------

HALF_LOG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
SETS = ("epa", "ll4", "dromics")
SET_SIZES = {
    "epa": (2, 3, 3, 4, 2, 3, 3, 4),
    "ll4": (4,),
    "dromics": (2, 3, 4, 5, 4, 4, 5, 4, 4),
}
PYBMDS_MODELS = (
    "linear",
    "polynomial_2",
    "power",
    "hill",
    "exponential_3",
    "exponential_5",
)
FAMILIES = {
    "L": ("linear",),
    "E": ("exponential",),
    "H": ("ll4",),
    "lGP": (
        "log_gauss_probit",
        "log_gauss_probit_monotone",
        "log_gauss_probit_symmetric",
    ),
    "GP": ("gauss_probit", "gauss_probit_monotone", "gauss_probit_symmetric"),
}
DROMICS_MODELS = {
    "linear": "L",
    "exponential": "E",
    "Hill": "H",
    "log-Gauss-probit": "lGP",
    "Gauss-probit": "GP",
}


@dataclass
class Samples:
    """A gene set's samples: doses, values (samples x genes) and the genes' truth."""

    name: str
    doses: np.ndarray
    values: np.ndarray
    genes: list[str]
    shapes: list[str]


@dataclass
class Means:
    """A cell type's curve data in a plug-in mode, from dose_response."""

    name: str
    variance: str
    fits: Any  # DoseCurveFits
    genes: list[str]


def axis() -> Any:
    from cytepool.dose_axes import DoseAxis

    return DoseAxis("dose", unit="ug/kg", metric="administered")


def simulate(name: str) -> tuple[Any, Any, Any]:
    """A series of the module documentation: its cube, truth and design."""
    from cytepool.design import Design
    from cytepool.simulated_dose_series import simulate_dose_series

    settings: dict[str, Any] = {
        "ladder": dict(
            doses=HALF_LOG,
            replicates=3,
            cell_types=["abundant", "depleted"],
            cells=[300.0, 200.0],
            n_genes=300,
            responsive=0.6,
            composition={"depleted": [1.0] * 8 + [0.03]},
            rng=20261220,
        ),
        "decades": dict(
            doses=[0.0, 0.01, 0.1, 1.0, 10.0],
            replicates=3,
            cell_types=1,
            cells=300.0,
            n_genes=200,
            responsive=0.6,
            rng=20261221,
        ),
        "four": dict(
            doses=[0.0, 0.1, 1.0, 10.0],
            replicates=2,
            cell_types=1,
            cells=300.0,
            n_genes=200,
            responsive=0.6,
            rng=20261222,
        ),
        "descriptive": dict(
            doses=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
            replicates=[3, 1, 1, 1, 1, 1],
            cell_types=1,
            cells=300.0,
            n_genes=200,
            responsive=0.6,
            rng=20261223,
        ),
        "sex": dict(
            doses=[0.0, 0.1, 1.0, 10.0, 30.0],
            replicates=4,
            cell_types=1,
            cells=300.0,
            n_genes=200,
            responsive=0.6,
            rng=20261224,
        ),
    }[name]
    doses = settings.pop("doses")
    cube, truth = simulate_dose_series(doses, **settings)
    formula = "~ 1"
    samples = cube.samples
    if name == "sex":
        ids = samples["sample_id"].tolist()
        samples = samples.with_columns(
            {"sex": ["F" if s.endswith(("_r1", "_r2")) else "M" for s in ids]}
        )
        formula = "~ sex"
    return cube, truth, Design(samples, formula)


def chosen_genes(truth: Any, cell_type: str, genes: list[str], count: int) -> list[int]:
    """Positions among ``genes`` of a spread of shapes: the responsive genes of each
    shape in turn, then flat ones, ``count`` in all."""
    table = truth.table(cell_type)
    shape_of = dict(zip(table["feature_id"].tolist(), table["shape"].tolist()))
    by_shape: dict[str, list[int]] = {}
    for position, gene in enumerate(genes):
        by_shape.setdefault(shape_of[gene], []).append(position)
    order = [s for s in by_shape if s != "flat"] + ["flat"]
    picked: list[int] = []
    while len(picked) < count and any(by_shape.get(s) for s in order):
        for s in order:
            if by_shape.get(s) and len(picked) < count:
                picked.append(by_shape[s].pop(0))
    return sorted(picked)


def constant_samples() -> list[Samples]:
    """The samples of the series fitted under constant variance: each gene's values
    in the cell type's fit by limma_voom, as dose_response refits them."""
    from cytepool.dose_responses import dose_response

    found = []
    for name, count in (("ladder", 12), ("decades", 8), ("four", 8)):
        cube, truth, design = simulate(name)
        result = dose_response(
            cube, design, dose=axis(), groups=cube.groups["group_id"][0]
        )
        group = result.groups[0]
        fit = result.fit(group)
        genes = fit.features["feature_id"].tolist()
        picked = chosen_genes(truth, group, genes, count)
        shape_of = dict(
            zip(
                truth.table(group)["feature_id"].tolist(),
                truth.table(group)["shape"].tolist(),
            )
        )
        found.append(
            Samples(
                name,
                np.asarray(fit.samples["dose"], dtype=float),
                np.asarray(fit.log2_cpm)[:, picked],
                [genes[p] for p in picked],
                [shape_of[genes[p]] for p in picked],
            )
        )
    return found


def shifted(samples: Samples) -> Samples:
    """The same genes moved up by 500 log2 units, for the question whether pybmds's
    optimizer depends on the response's level."""
    return Samples(
        samples.name + "+500",
        samples.doses,
        samples.values + 500.0,
        samples.genes,
        samples.shapes,
    )


def plugin_means() -> list[Means]:
    """Curve data of the weighted and moderated modes from dose_response: the ladder's
    two cell types and the series with sex in the design."""
    from cytepool.dose_responses import dose_response

    found = []
    for name in ("ladder", "sex"):
        cube, truth, design = simulate(name)
        for variance in ("weighted", "moderated"):
            result = dose_response(
                cube, design, dose=axis(), models="dromics", variance=variance
            )
            for group in result.groups:
                fits = result.curve_fits(group)
                genes = result.table(group, "curves")["feature_id"][:: len(fits.models)]
                found.append(Means(f"{name} {group}", variance, fits, genes.tolist()))
    return found


# ---------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------

LIKELIHOOD = 1e-8  # the residual variance relative, or Q / (K s^2), the scale-free gap
EXACT = 1e-10
CLOSE = 1e-6
PARAMETER_BOUND = 10.0  # standard errors per sqrt(2 max(gap in LL, 1e-12))
# The largest condition number of a fit's Jacobian, columns scaled to unit length, at
# which its parameters count as identified and are compared.
IDENTIFIED = 1e6
TURN_DOSE = 1e-3  # relative, from the convergence of DRomics' nls fits


@dataclass
class Rows:
    """The table's rows: check, reference, value, tolerance, result; and the worst
    case of each failed row."""

    rows: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    failures: int = 0
    worst: dict[str, str] = field(default_factory=dict)

    def check(
        self,
        check: str,
        reference: str,
        value: float,
        tolerance: float,
        count: int,
        worst: str = "",
    ) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        if not passed:
            self.worst[check] = worst
        self.rows.append(
            (
                f"{check} ({count:,})",
                reference,
                f"{value:.1e}",
                f"{tolerance:g}",
                "pass" if passed else "FAIL",
            )
        )

    def identical(
        self, check: str, reference: str, mismatches: int, count: int
    ) -> None:
        self.failures += mismatches > 0
        self.rows.append(
            (
                f"{check} ({count:,})",
                reference,
                f"{mismatches} mismatches",
                "0",
                "pass" if mismatches == 0 else "FAIL",
            )
        )

    def note(self, check: str, reference: str, value: str, tier: str) -> None:
        self.rows.append((check, reference, value, "n/a", tier))

    def print(self) -> None:
        print("| check | reference | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")
        for check, worst in self.worst.items():
            print(f"differs: {check}: {worst}")


def model_parameters(fits: Any, m: int) -> np.ndarray:
    """A model's parameters of every gene, in the model's own order."""
    from cytepool.stats._dose_curve_models import model_of

    spec = model_of(fits.model_set, fits.models[m])
    positions = [fits.parameter_names.index(name) for name in spec.parameters]
    return np.asarray(fits.parameters[:, m][:, positions])


def evaluated(model: str, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A model's mean at reported parameters, by the documented formula."""
    if model == "exponential":
        b, d, e = theta
        return d + b * np.expm1(x / e)
    return documented_mean(model, theta, x)


def documented_name(model_set: str, model: str) -> str:
    return "dromics_linear" if (model_set == "dromics" and model == "linear") else model


def q_of(fits: Any, g: int, m: int) -> float:
    """Q of a fit at its reported parameters, by the documented formula, whitened by
    the means' covariance."""
    model = documented_name(fits.model_set, fits.models[m])
    theta = model_parameters(fits, m)[g]
    root = np.linalg.inv(np.linalg.cholesky(np.asarray(fits.covariance[g])))
    r = root @ (fits.means[g] - evaluated(model, theta, np.asarray(fits.doses)))
    return float(r @ r)


def within_sum(fits: Any, g: int) -> float:
    """The within-group sum of squares under constant variance, from the pooled
    variance and its degrees of freedom; 0 without replicates."""
    df = float(fits.df[g])
    return float(fits.covariance_scale[g]) * df if df > 0 else 0.0


def likelihood_of(fits: Any, g: int, q: float) -> float:
    """The documented log-likelihood of a fit of objective q."""
    n = float(np.sum(fits.n_samples))
    k = fits.doses.size
    if fits.variance == "constant":
        return -0.5 * n * (math.log(2 * math.pi * (within_sum(fits, g) + q) / n) + 1)
    s2 = float(fits.covariance_scale[g])
    _, logdet = np.linalg.slogdet(np.asarray(fits.covariance[g]))
    return (
        -0.5 * k * math.log(2 * math.pi)
        - 0.5 * (k * math.log(s2) + logdet)
        - q / (2 * s2)
    )


def gap(fits: Any, g: int, q_ours: float, q_reference: float) -> float:
    """How far CytePool's maximum lies below the reference's, scale-free: the residual
    variance relative under constant variance, (Q - Q_ref) / (K s^2) otherwise;
    negative where CytePool's is the larger."""
    if fits.variance == "constant":
        return (q_ours - q_reference) / (within_sum(fits, g) + q_reference)
    return (q_ours - q_reference) / (fits.doses.size * float(fits.covariance_scale[g]))


def jacobian(fits: Any, g: int, m: int) -> np.ndarray:
    """The Jacobian of a fit's documented mean at its reported parameters, by central
    differences, whitened by the means' covariance."""
    model = documented_name(fits.model_set, fits.models[m])
    theta = model_parameters(fits, m)[g]
    x = np.asarray(fits.doses)
    root = np.linalg.inv(np.linalg.cholesky(np.asarray(fits.covariance[g])))
    jac = np.empty((x.size, theta.size))
    for j in range(theta.size):
        h = 1e-6 * max(abs(theta[j]), 1e-3)
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        jac[:, j] = (
            root @ (evaluated(model, up, x) - evaluated(model, down, x)) / (2 * h)
        )
    return jac


def identified(fits: Any, g: int, m: int) -> bool:
    """Whether the data identify a fit's parameters: the condition number of its
    Jacobian, columns scaled to unit length, is at most IDENTIFIED. Standard errors,
    and comparisons in their units, mean nothing along a direction the data do not
    determine, such as the exponent of a power curve whose change rests on the top
    dose alone."""
    jac = jacobian(fits, g, m)
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(np.isfinite(jac)) or np.any(norms == 0):
        return False
    s = np.linalg.svd(jac / norms, compute_uv=False)
    return bool(s[-1] > 0 and s[0] / s[-1] <= IDENTIFIED)


def numeric_errors(fits: Any, g: int, m: int) -> np.ndarray:
    """Standard errors of a fit's reported parameters from the Jacobian of the
    documented formula by central differences, with the residual variance."""
    theta = model_parameters(fits, m)[g]
    jac = jacobian(fits, g, m)
    n = float(np.sum(fits.n_samples))
    s2 = (within_sum(fits, g) + q_of(fits, g, m)) / max(n - theta.size, 1.0)
    try:
        covariance = np.linalg.inv(jac.T @ jac) * s2
    except np.linalg.LinAlgError:
        return np.full(theta.size, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.diag(covariance))


def parameter_distance(
    ours: np.ndarray, reference: np.ndarray, errors: np.ndarray, log_gap: float
) -> float:
    """The largest |theta - theta_ref| over SE sqrt(2 max(gap, 1e-12)): a quadratic
    log-likelihood within ``log_gap`` of its maximum puts each parameter within
    sqrt(2 log_gap) standard errors of it."""
    scale = errors * math.sqrt(2.0 * max(log_gap, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(ours - reference) / scale
    ratio = ratio[np.isfinite(ratio)]
    return float(ratio.max()) if ratio.size else 0.0


def p_distance(ours: float, reference: float) -> float:
    """The difference of two p-values relative to p max(1, |log p|)."""
    if np.isnan(ours) and np.isnan(reference):
        return 0.0
    scale = max(reference * max(1.0, abs(math.log(max(reference, 1e-300)))), 1e-300)
    return abs(ours - reference) / scale


@dataclass
class Case:
    """Fits of one set to some genes, with the genes the references see."""

    label: str
    fits: Any  # DoseCurveFits
    genes: list[int]  # rows of the fits the references see
    samples: Samples | None = None  # under constant variance from samples


def constant_cases(found: list[Samples]) -> list[Case]:
    from cytepool.stats.dose_curves import _no_model_fits, fit_dose_curves

    return [
        Case(
            f"{s.name} {model_set}",
            fit_dose_curves(s.doses, s.values, models=model_set),
            list(range(len(s.genes))),
            s,
        )
        for s in found
        for model_set in SETS
        # A set whose every model needs more dose groups than the series has.
        if not _no_model_fits(model_set, np.unique(s.doses).size)
    ]


def plugin_cases(per_case: int) -> tuple[list[Case], dict[str, Any]]:
    """The weighted and moderated fits of the three sets on the ladder's cell types
    and on the series with sex in the design, and the inputs of limma's jobs."""
    from cytepool.dose_responses import dose_response
    from cytepool.differential_expression import limma_trend

    cases, limma_jobs = [], {}
    for name in ("ladder", "sex"):
        cube, truth, design = simulate(name)
        for model_set in SETS:
            for variance in ("weighted", "moderated"):
                result = dose_response(
                    cube, design, dose=axis(), models=model_set, variance=variance
                )
                for group in result.groups:
                    fits = result.curve_fits(group)
                    genes = result.table(group, "curves")["feature_id"][
                        :: len(fits.models)
                    ].tolist()
                    picked = chosen_genes(truth, group, genes, per_case)
                    cases.append(
                        Case(f"{name} {group} {variance} {model_set}", fits, picked)
                    )
                    if model_set == "dromics" and variance == "weighted":
                        limma_jobs[f"{name} {group} voom"] = limma_job(
                            cube, design, result, group, "voom", fits, genes
                        )
        # limma-trend, whose F-test of the lack-of-fit contrasts limma computes exactly.
        result = dose_response(
            cube,
            design,
            dose=axis(),
            models="epa",
            variance="moderated",
            analysis=limma_trend,
        )
        for group in result.groups:
            fits = result.curve_fits(group)
            genes = result.table(group, "curves")["feature_id"][
                :: len(fits.models)
            ].tolist()
            cases.append(Case(f"{name} {group} moderated trend epa", fits, []))
            limma_jobs[f"{name} {group} trend"] = limma_job(
                cube, design, result, group, "trend", fits, genes
            )
    return cases, limma_jobs


def limma_job(
    cube: Any,
    design: Any,
    result: Any,
    group: str,
    analysis: str,
    fits: Any,
    genes: list[str],
) -> dict[str, Any]:
    """What R needs to fit a cell type's dose groups as dose_response does: the
    counts and design of the samples used, the contrasts of the vehicle's mean and of
    each dose, and contrasts on the means that span the lack of fit of a line."""
    rows = result.design_report.samples.filter(group_id=group)
    used = np.asarray(rows["used"], dtype=bool)
    ids = rows["sample_id"][used].tolist()
    position = {s: i for i, s in enumerate(cube.samples["sample_id"].tolist())}
    cube_rows = sorted(position[s] for s in ids)
    g = cube.groups["group_id"].tolist().index(group)
    counts = np.asarray(cube.layer("counts", g))[cube_rows]
    samples = design.samples
    sample_row = {s: i for i, s in enumerate(samples["sample_id"].tolist())}
    design_rows = [sample_row[cube.samples["sample_id"][r]] for r in cube_rows]
    doses = np.asarray(samples["dose"], dtype=float)[design_rows]
    levels = np.asarray(fits.doses)
    others = design.matrix[design_rows]
    indicators = np.column_stack(
        [(doses == level).astype(float) for level in levels[1:]]
    )
    x = np.column_stack([others, indicators])
    vehicle = np.r_[others.mean(axis=0), np.zeros(levels.size - 1)]
    contrasts = np.column_stack(
        [
            vehicle,
            np.r_[
                np.zeros((others.shape[1], levels.size - 1)), np.eye(levels.size - 1)
            ],
        ]
    )
    # Contrasts on the K means that span the null space of [1, x] at the doses.
    line = np.column_stack([np.ones(levels.size), levels])
    q, _ = np.linalg.qr(line, mode="complete")
    lack = q[:, 2:]
    return {
        "analysis": analysis,
        "what": "fit",
        "n_samples": len(cube_rows),
        "counts": counts.tolist(),
        "design": x.tolist(),
        "contrasts": contrasts.tolist(),
        "lack_of_fit": lack.tolist(),
        "genes": genes,
        "feature_ids": cube.features["feature_id"].tolist(),
        "fits": fits,
    }


# Genes that turn, or do not, by construction, at 5, 6 and 9 dose groups.
TURN_LEVELS = {
    5: [0.0, 0.1, 1.0, 10.0, 30.0],
    6: [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
    9: HALF_LOG,
}


def turn_shapes(levels: np.ndarray) -> np.ndarray:
    """Means at the doses, genes x doses, from a level of 6 log2 units: one dose group
    raised by 1.5 at each treated dose but the top, a spike that a bell can fit; five
    bells of height 1.5 spread over about three groups on log dose; three rises at
    the top dose alone, whose true means peak there and so do not turn; and two flat
    genes."""
    positions = np.log10(levels[1:])
    width = 1.2 * float(np.median(np.diff(positions)))
    rows = []
    for k in range(1, levels.size - 1):
        rows.append(np.where(np.arange(levels.size) == k, 7.5, 6.0))
    for centre in np.linspace(positions[1], positions[-2], 5):
        bell = 1.5 * np.exp(-0.5 * ((positions - centre) / width) ** 2)
        rows.append(np.r_[6.0, 6.0 + bell])
    for _ in range(3):
        rows.append(np.where(np.arange(levels.size) == levels.size - 1, 7.5, 6.0))
    rows += [np.full(levels.size, 6.0)] * 2
    return np.array(rows)


def constructed_turn_cases(plugin: list[Case]) -> list[Case]:
    """The "dromics" fits of genes that turn, or do not, by construction: under
    constant variance, samples at 5, 6 and 9 dose groups x 3 with an SD of 0.15; in
    the weighted mode, dose-group means drawn about the same shapes with the
    covariance and residual variance of the first genes of the ladder's and the sex
    series' fits, at 9 and 5 dose groups."""
    from cytepool.stats.dose_curves import _CurveData, _fit_means, fit_dose_curves

    cases = []
    for k, levels in TURN_LEVELS.items():
        shapes = turn_shapes(np.asarray(levels))
        rng = np.random.default_rng(20261226 + k)
        doses = np.repeat(levels, 3)
        values = np.repeat(shapes.T, 3, axis=0) + rng.normal(
            0.0, 0.15, (doses.size, len(shapes))
        )
        samples = Samples(
            f"turns{k}",
            doses,
            values,
            [f"t{g}" for g in range(len(shapes))],
            ["constructed"] * len(shapes),
        )
        fits = fit_dose_curves(doses, values, models="dromics")
        cases.append(Case(f"turns{k} dromics", fits, list(range(len(shapes))), samples))
    for label in (
        "ladder abundant weighted dromics",
        "sex cell_type_1 weighted dromics",
    ):
        source = next(c.fits for c in plugin if c.label == label)
        levels = np.asarray(source.doses)
        shapes = turn_shapes(levels)
        n = len(shapes)
        rng = np.random.default_rng(20261236 + levels.size)
        covariance = np.asarray(source.covariance[:n])
        scale = np.asarray(source.covariance_scale[:n])
        noise = np.einsum(
            "gij,gj->gi",
            np.linalg.cholesky(covariance),
            rng.normal(size=(n, levels.size)),
        )
        data = _CurveData(
            "weighted",
            levels,
            np.asarray(source.n_samples),
            shapes + noise * np.sqrt(scale)[:, None],
            covariance,
            scale,
            np.asarray(source.df[:n]),
            None,
        )
        cases.append(
            Case(f"turns {label}", _fit_means(data, "dromics"), list(range(n)))
        )
    return cases


def descriptive_job() -> tuple[Any, dict[str, Any]]:
    """The cell type without replicates at its doses, its curves and its R job."""
    from cytepool.dose_responses import dose_response

    cube, _, design = simulate("descriptive")
    result = dose_response(cube, design, dose=axis(), models="epa")
    group = result.groups[0]
    fits = result.curve_fits(group)
    genes = result.table(group, "curves")["feature_id"][:: len(fits.models)].tolist()
    job = limma_job(cube, design, result, group, "voom", fits, genes)
    job["what"] = "values"
    return fits, job


def scipy_tasks(cases: list[Case], starts: int) -> list[dict[str, Any]]:
    tasks = []
    for c, case in enumerate(cases):
        fits = case.fits
        for m, model in enumerate(fits.models):
            name = documented_name(fits.model_set, model)
            for g in case.genes:
                if fits.not_fitted_reason[g, m]:
                    continue
                root = np.linalg.inv(np.linalg.cholesky(np.asarray(fits.covariance[g])))
                tasks.append(
                    {
                        "key": f"{c}|{m}|{g}",
                        "model": name,
                        "doses": np.asarray(fits.doses).tolist(),
                        "means": np.asarray(fits.means[g]).tolist(),
                        "root": root.tolist(),
                        "starts": starts,
                        "seed": 20261230 + len(tasks),
                    }
                )
    return tasks


def compare_scipy(
    rows: Rows, cases: list[Case], results: dict[str, Any]
) -> dict[str, float]:
    """CytePool's maximum against scipy's best of 200 starts, per mode and set, and
    the log-likelihood of the reported parameters by the documented formula."""
    worst: dict[str, float] = {}
    where: dict[str, str] = {}
    counts: dict[str, int] = {}
    formula, n_formula = 0.0, 0
    for key, found in results.items():
        c, m, g = map(int, key.split("|"))
        fits = cases[c].fits
        q_ours = q_of(fits, g, m)
        label = f"{fits.variance} variance, {fits.model_set}"
        value = gap(fits, g, q_ours, found["q"])
        if value > worst.get(label, -np.inf):
            worst[label] = value
            where[label] = (
                f"{cases[c].label}, {fits.models[m]}, gene {g}: Q {q_ours:.12g} against "
                f"{found['q']:.12g}; CytePool {model_parameters(fits, m)[g].tolist()}, "
                f"scipy {found['theta']}"
            )
        counts[label] = counts.get(label, 0) + 1
        scale = (
            float(np.sum(fits.n_samples))
            if fits.variance == "constant"
            else fits.doses.size
        )
        formula = max(
            formula,
            2 * abs(fits.log_likelihood[g, m] - likelihood_of(fits, g, q_ours)) / scale,
        )
        n_formula += 1
    for label in sorted(worst):
        rows.check(
            f"{label}: maximum not below scipy's best of 200 starts, gap scale-free",
            "scipy 1.18.1 least_squares (trf, tolerances 1e-15)",
            worst[label],
            LIKELIHOOD,
            counts[label],
            where[label],
        )
    rows.check(
        "log-likelihood of the reported parameters by the documented formula, "
        "2|dLL| / n",
        "documented means and likelihoods",
        formula,
        EXACT,
        n_formula,
    )
    return worst


def compare_pybmds(
    rows: Rows, cases: list[Case], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """CytePool's "epa" fits under constant variance against pybmds 25.2: each model
    run increasing and decreasing, the larger maximum kept."""
    from cytepool.stats._dose_curve_models import model_of
    from cytepool.stats.distributions import chi2_sf_scalar, f_isf_scalar

    by_key = {r["key"]: r for r in results}
    largest_gap, n_gap, gap_where = -np.inf, 0, ""
    beyond: dict[str, list[str]] = {}
    lower: dict[str, list[float]] = {}
    shifted_lower: dict[str, list[float]] = {}
    params, n_params, params_where = 0.0, 0, ""
    penalty_mismatch, n_penalty = 0, 0
    test_p, n_test, df_mismatch = 0.0, 0, 0
    residual, n_residual, residual_where = 0.0, 0, ""
    direction_matters = 0
    failed_runs, n_runs = 0, 0
    failed_both: dict[str, int] = {}
    for case in cases:
        fits, s = case.fits, case.samples
        if fits.model_set != "epa" or s is None:
            continue
        n = float(np.sum(fits.n_samples))
        for m, model in enumerate(fits.models):
            if model not in PYBMDS_MODELS:
                continue
            spec = model_of("epa", model)
            for g in case.genes:
                runs = {
                    d: by_key.get(f"{s.name}|{model}|{g}|{d}") for d in (True, False)
                }
                if fits.not_fitted_reason[g, m]:
                    continue
                # pybmds reports a fit that did not complete with no log-likelihood,
                # or -9999.
                usable = {
                    d: r
                    for d, r in runs.items()
                    if r is not None
                    and r["log_likelihood"] is not None
                    and r["log_likelihood"] > -9e3
                }
                if not s.name.endswith("+500"):
                    n_runs += len(runs)
                    failed_runs += len(runs) - len(usable)
                    if not usable:
                        failed_both[model] = failed_both.get(model, 0) + 1
                runs = usable
                if not runs:
                    continue
                increasing = max(runs, key=lambda d: runs[d]["log_likelihood"])
                best = runs[increasing]
                if (
                    len(runs) == 2
                    and abs(
                        runs[True]["log_likelihood"] - runs[False]["log_likelihood"]
                    )
                    > 1e-6
                ):
                    direction_matters += 1
                ll_ours, ll_ref = fits.log_likelihood[g, m], best["log_likelihood"]
                value = math.expm1(2 * (ll_ref - ll_ours) / n)
                target = shifted_lower if s.name.endswith("+500") else lower
                if value < -LIKELIHOOD:
                    target.setdefault(model, []).append(ll_ours - ll_ref)
                if s.name.endswith("+500"):
                    continue
                # pybmds's exponential 3 has b >= 0 and its direction as a setting.
                sign = 1.0 if increasing else -1.0
                ref = dict(zip(best["names"], best["values"]))
                reference = np.array(
                    [
                        ref["b"] * sign
                        if (nm == "b" and model == "exponential_3")
                        else ref[nm]
                        for nm in spec.parameters
                    ]
                )
                ours = model_parameters(fits, m)[g]
                if value > LIKELIHOOD and not within_bounds(
                    model, reference, np.asarray(fits.doses)
                ):
                    # pybmds's maximum lies beyond CytePool's bounds.
                    beyond.setdefault(model, []).append(
                        f"{s.name} gene {g}: {dict(zip(spec.parameters, reference))}"
                    )
                    continue
                if value > largest_gap:
                    gap_where = (
                        f"{s.name}, {model}, gene {g}: log-likelihood {ll_ours:.10g} "
                        f"against {ll_ref:.10g}; CytePool {ours.tolist()}, pybmds "
                        f"{reference.tolist()}, bounded {best['bounded']}"
                    )
                largest_gap, n_gap = max(largest_gap, value), n_gap + 1
                if abs(value) > LIKELIHOOD:
                    continue
                penalty_mismatch += not math.isclose(
                    best["aic"] + 2 * ll_ref,
                    fits.aic[g, m] + 2 * ll_ours,
                    rel_tol=1e-12,
                )
                n_penalty += 1
                # Test 4 is the likelihood ratio of CytePool's F-test, n log(1 + (K - p)
                # F / (n - K)), on a chi-square of K - p degrees of freedom.
                spare = fits.doses.size - len(ours)
                within_df = n - fits.doses.size
                f = f_isf_scalar(
                    float(fits.lack_of_fit_p_value[g, m]), spare, within_df
                )
                test4 = chi2_sf_scalar(n * math.log1p(spare * f / within_df), spare)
                test_p = max(test_p, p_distance(test4, best["test4_p"]))
                df_mismatch += best["test4_df"] != spare
                n_test += 1
                # Two maxima within 1e-8 of the residual variance put each scaled
                # residual within 2 sqrt(n 1e-8) of the other's.
                ref_residual = float(np.max(np.abs(best["scaled_residuals"])))
                spread = abs(fits.max_abs_scaled_residual[g, m] - ref_residual) / (
                    2.0 * math.sqrt(n * LIKELIHOOD)
                )
                if spread > residual:
                    residual_where = (
                        f"{s.name}, {model}, gene {g}: "
                        f"{fits.max_abs_scaled_residual[g, m]:.10g} against "
                        f"{ref_residual:.10g}"
                    )
                residual, n_residual = max(residual, spread), n_residual + 1
                if (
                    fits.parameters_at_bounds[g, m]
                    or any(best["bounded"])
                    or not identified(fits, g, m)
                ):
                    continue
                se = dict(zip(best["names"], best["se"]))
                errors = np.array(
                    [se.get(nm, np.nan) for nm in spec.parameters], dtype=float
                )
                # Where pybmds reports no finite, positive standard error for a
                # parameter, its covariance is degenerate, as when a flat gene's
                # power curve leaves n unidentified: the fit is not compared.
                if not (np.isfinite(errors).all() and (errors > 0).all()):
                    continue
                distance = parameter_distance(
                    ours, reference, errors, abs(ll_ours - ll_ref)
                )
                if distance > params:
                    params_where = (
                        f"{s.name}, {model}, gene {g}: CytePool {ours.tolist()}, pybmds "
                        f"{reference.tolist()}, SE {errors.tolist()}, dLL "
                        f"{ll_ours - ll_ref:.3g}"
                    )
                params, n_params = max(params, distance), n_params + 1
    rows.check(
        "constant variance, epa: maximum not below pybmds's best of both directions, "
        "residual variance relative",
        "pybmds 25.2, normal errors of constant variance, default bounds",
        largest_gap,
        LIKELIHOOD,
        n_gap,
        gap_where,
    )
    rows.note(
        f"pybmds completes no fit in {failed_runs} of {n_runs} runs, and in neither "
        f"direction for {sum(failed_both.values())} fits ("
        + ", ".join(f"{k} {v}" for k, v in sorted(failed_both.items()))
        + ")",
        "pybmds 25.2",
        f"{sum(failed_both.values())}",
        "deliberate difference",
    )
    n_beyond = sum(len(v) for v in beyond.values())
    rows.note(
        f"pybmds's maximum lies beyond CytePool's bounds in {n_beyond} fits ("
        + "; ".join(f"{k}: " + ", ".join(v) for k, v in sorted(beyond.items()))
        + ")",
        "pybmds 25.2",
        f"{n_beyond}",
        "deliberate difference",
    )
    total = sum(len(v) for v in lower.values())
    largest = max((max(v) for v in lower.values()), default=0.0)
    rows.note(
        f"pybmds's maximum lies below CytePool's in {total} fits ("
        + ", ".join(f"{k} {len(v)}" for k, v in sorted(lower.items()))
        + f"), by up to {largest:.3g} log-likelihood units",
        "pybmds 25.2",
        f"{total} of {n_gap}",
        "deliberate difference",
    )
    rows.check(
        "the same fits where both reach the maximum: parameters off every bound and "
        "identified, with pybmds's standard errors all finite, |d theta| / (SE "
        "sqrt(2 max(dLL, 1e-12)))",
        "pybmds 25.2",
        params,
        PARAMETER_BOUND,
        n_params,
        params_where,
    )
    rows.identical(
        "AIC: every parameter counted, AIC + 2 LL",
        "pybmds count_all_parameters_on_boundary=True",
        penalty_mismatch,
        n_penalty,
    )
    rows.identical(
        "lack of fit: degrees of freedom of test 4, K - p",
        "pybmds test 4",
        df_mismatch,
        n_test,
    )
    rows.check(
        "lack of fit: test 4 from CytePool's F, n log(1 + (K - p) F / (n - K)) on a chi-square of K - p degrees of freedom, p relative to p max(1, |log p|)",
        "pybmds test 4",
        test_p,
        CLOSE,
        n_test,
    )
    rows.check(
        "largest absolute scaled residual, in units of 2 sqrt(n 1e-8), the spread "
        "that two maxima within 1e-8 allow",
        "pybmds goodness of fit",
        residual,
        1.0,
        n_residual,
        residual_where,
    )
    shifted_total = sum(len(v) for v in shifted_lower.values())
    rows.note(
        f"pybmds with the responses moved up by 500: its maximum lies below CytePool's "
        f"in {shifted_total} fits ("
        + ", ".join(f"{k} {len(v)}" for k, v in sorted(shifted_lower.items()))
        + ")",
        "pybmds 25.2",
        f"{shifted_total}",
        "information",
    )
    rows.note(
        f"pybmds's maximum depends on the modelling direction in {direction_matters} "
        "fits",
        "pybmds 25.2, is_increasing True and False",
        f"{direction_matters}",
        "information",
    )
    return {
        "lower": {k: len(v) for k, v in lower.items()},
        "shifted_lower": {k: len(v) for k, v in shifted_lower.items()},
    }


def compare_lack_of_fit(rows: Rows, cases: list[Case], dose_r: dict[str, Any]) -> None:
    """The F-test of lack of fit under constant variance against R's anova of the
    line, and of the quadratic where its restriction to one sign does not bind,
    against free means of the dose groups, on each series' samples."""
    largest, n = 0.0, 0
    for case in cases:
        fits, s = case.fits, case.samples
        if s is None or s.name not in dose_r["series"] or fits.model_set == "ll4":
            continue
        found = dose_r["series"][s.name]
        pairs = [("linear", found["lof_linear"])]
        if fits.model_set == "epa":
            pairs.append(("polynomial_2", found["lof_quadratic"]))
        for model, reference in pairs:
            m = fits.models.index(model)
            for g in case.genes:
                if model == "polynomial_2" and fits.parameters_at_bounds[g, m]:
                    continue
                largest = max(
                    largest,
                    p_distance(
                        float(fits.lack_of_fit_p_value[g, m]), float(reference[g])
                    ),
                )
                n += 1
    rows.check(
        "constant variance: F-test of lack of fit of a line, and of the quadratic where its restriction does not bind, p relative to p max(1, |log p|)",
        "R 4.4.3 anova(lm(y ~ dose), lm(y ~ factor(dose)))",
        largest,
        EXACT,
        n,
    )


def drc_parameters(coef: list[float]) -> np.ndarray:
    """drc's LL.4 estimates (b, c, d, e) in CytePool's letters: b > 0 with d the level
    at dose 0, so (-b, d, c, e) when drc's b < 0."""
    b, c, d, e = coef
    return np.array([b, c, d, e]) if b > 0 else np.array([-b, d, c, e])


def within_bounds(model: str, theta: np.ndarray, doses: np.ndarray) -> bool:
    lower, upper = documented_bounds(model, doses)
    return bool(np.all(theta >= lower) & np.all(theta <= upper))


def compare_drc(rows: Rows, cases: list[Case], dose_r: dict[str, Any]) -> None:
    largest, n, params, n_params = -np.inf, 0, 0.0, 0
    deliberate: dict[str, int] = {
        "failed": 0,
        "lower": 0,
        "beyond CytePool's bounds": 0,
    }
    for case in cases:
        fits, s = case.fits, case.samples
        if fits.model_set != "ll4" or s is None or s.name.endswith("+500"):
            continue
        nsum = float(np.sum(fits.n_samples))
        for g in case.genes:
            found = dose_r["series"][s.name]["drc"][g]
            if found["failed"]:
                deliberate["failed"] += 1
                continue
            reference = drc_parameters(found["coef"])
            ll_ours, ll_ref = fits.log_likelihood[g, 0], found["loglik"]
            value = math.expm1(2 * (ll_ref - ll_ours) / nsum)
            if value > LIKELIHOOD and not within_bounds(
                "ll4", reference, np.asarray(fits.doses)
            ):
                deliberate["beyond CytePool's bounds"] += 1
                continue
            if value < -LIKELIHOOD:
                deliberate["lower"] += 1
            largest, n = max(largest, value), n + 1
            if (
                abs(value) <= LIKELIHOOD
                and not fits.parameters_at_bounds[g, 0]
                and identified(fits, g, 0)
            ):
                se = np.asarray(found["se"], dtype=float)
                errors = se if reference[0] == found["coef"][0] else se[[0, 2, 1, 3]]
                params = max(
                    params,
                    parameter_distance(
                        model_parameters(fits, 0)[g],
                        reference,
                        errors,
                        abs(ll_ours - ll_ref),
                    ),
                )
                n_params += 1
    rows.check(
        "constant variance, ll4: maximum not below drc's, residual variance relative",
        "drc 4.0-0 LL.4, drmc(relTol = 1e-12, maxIt = 10000)",
        largest,
        LIKELIHOOD,
        n,
    )
    rows.note(
        "drc: " + ", ".join(f"{k} {v}" for k, v in deliberate.items()),
        "drc 4.0-0",
        f"{sum(deliberate.values())}",
        "deliberate difference",
    )
    rows.check(
        "ll4 parameters where both reach the maximum, off every bound and identified, drc's (b, c, d, e) as (-b, d, c, e) when b < 0: |d theta| / (SE sqrt(2 max(dLL, 1e-12)))",
        "drc 4.0-0 coef, vcov",
        params,
        PARAMETER_BOUND,
        n_params,
    )


def family_aicc(fits: Any, g: int) -> dict[str, float]:
    """CytePool's lowest AICc in each of DRomics' families."""
    out = {}
    for family, members in FAMILIES.items():
        values = [
            fits.aicc[g, fits.models.index(m)] for m in members if m in fits.models
        ]
        values = [v for v in values if np.isfinite(v)]
        out[family] = min(values) if values else np.inf
    return out


def dromics_form(row: dict[str, Any]) -> str:
    """CytePool's name of the model DRomics chose for an item."""
    family = DROMICS_MODELS[row["model"]]
    if family in ("L", "E", "H"):
        return FAMILIES[family][0]
    prefix = "log_gauss_probit" if family == "lGP" else "gauss_probit"
    if row["nbpar"] == 5:
        return prefix
    if row["f"] == 0:
        return prefix + "_monotone"
    return prefix + "_symmetric"


def compare_dromics(
    rows: Rows, cases: list[Case], dose_r: dict[str, Any]
) -> list[dict[str, Any]]:
    """CytePool's "dromics" fits under constant variance against DRomics' drcfit:
    the lowest AICc of each family, the chosen model's parameters, and where a bell or
    U turns."""
    from cytepool.stats._dose_curve_models import internal, model_of, turn_doses

    largest, n = -np.inf, 0
    params, n_params, turns, n_turns = 0.0, 0, 0.0, 0
    deliberate = {
        "DRomics' fit failed (Inf)": 0,
        "DRomics lower": 0,
        "DRomics higher beyond CytePool's bounds": 0,
        "DRomics higher, parameters not reported": 0,
    }
    matched = []
    unreported: list[str] = []
    for case in cases:
        fits, s = case.fits, case.samples
        if fits.model_set != "dromics" or s is None or s.name.endswith("+500"):
            continue
        n_total = float(np.sum(fits.n_samples))
        found = dose_r["series"][s.name]
        fitres = {row["id"]: row for row in found["fitres"]}
        criteria_of = {row["id"]: row for row in found["criteria"]}
        for item, row in fitres.items():
            # The criterion of the model DRomics chose is the lowest of its item's: the
            # rows of the criteria follow the items.
            chosen = criteria_of[item][f"InfoCrit.{DROMICS_MODELS[row['model']]}"]
            others = [
                criteria_of[item][f"InfoCrit.{family}"]
                for family in FAMILIES
                if criteria_of[item][f"InfoCrit.{family}"] is not None
            ]
            if chosen is None or chosen > min(others) + 1e-9:
                raise ValueError(
                    f"DRomics' criteria of {item} do not follow its chosen model "
                    f"{row['model']}: the criteria rows are not in the items' order"
                )
        for criteria in found["criteria"]:
            g = int(criteria["id"][1:])
            ours = family_aicc(fits, g)
            best_row = fitres.get(criteria["id"])
            for family in FAMILIES:
                ref = criteria[f"InfoCrit.{family}"]
                if ref is None or not np.isfinite(ref) or not np.isfinite(ours[family]):
                    deliberate["DRomics' fit failed (Inf)"] += (
                        ref is None or not np.isfinite(ref)
                    )
                    continue
                # DRomics prints its criteria to 2 decimals.
                value = (ours[family] - ref - 0.005) / n_total
                if value > LIKELIHOOD:
                    chosen = (
                        best_row is not None
                        and DROMICS_MODELS[best_row["model"]] == family
                    )
                    if chosen:
                        form = dromics_form(best_row)
                        spec_names = model_parameters_names(form)
                        theta = np.array(
                            [best_row[nm] for nm in spec_names], dtype=float
                        )
                        if not within_bounds(form, theta, np.asarray(fits.doses)):
                            deliberate["DRomics higher beyond CytePool's bounds"] += 1
                            continue
                    elif family != "L":
                        deliberate["DRomics higher, parameters not reported"] += 1
                        unreported.append(
                            f"{s.name} gene {g} {family}: AICc {ours[family]:.2f} "
                            f"against {ref:.2f}"
                        )
                        continue
                # Lower beyond the rounding of its printed criterion.
                if (ours[family] - ref + 0.005) / n_total < -LIKELIHOOD:
                    deliberate["DRomics lower"] += 1
                largest, n = max(largest, value), n + 1
            if best_row is None:
                continue
            form = dromics_form(best_row)
            if best_row.get("trend") in ("bell", "U"):
                # The turn at DRomics' own estimates, by CytePool's closed form.
                spec = model_of("dromics", form)
                theta = np.array([best_row[nm] for nm in spec.parameters], dtype=float)
                x_ours = float(
                    turn_doses(
                        spec.kind,
                        internal(spec.kind, theta)[None],
                        float(fits.doses[-1]),
                    )[0]
                )
                x_ref = float(best_row["xextrem"])
                # A turn that CytePool's form puts outside the doses is a miss.
                difference = (
                    abs(x_ours - x_ref) / x_ref if np.isfinite(x_ours) else math.inf
                )
                turns = max(turns, difference)
                n_turns += 1
            if form not in fits.models:
                continue
            m = fits.models.index(form)
            spec_names = model_parameters_names(form)
            reference = np.array([best_row[nm] for nm in spec_names], dtype=float)
            ll_gap = abs(fits.aicc[g, m] - best_row_aicc(criteria, form)) / 2
            if (
                ll_gap * 2 / n_total > LIKELIHOOD + 0.005 / n_total
                or fits.parameters_at_bounds[g, m]
                or not identified(fits, g, m)
            ):
                continue
            params = max(
                params,
                parameter_distance(
                    model_parameters(fits, m)[g],
                    reference,
                    numeric_errors(fits, g, m),
                    max(ll_gap, 0.0025),
                ),
            )
            n_params += 1
            matched.append(
                {"series": s.name, "gene": g, "model": form, "row": best_row}
            )
    rows.check(
        "constant variance, dromics: lowest AICc of each family not above DRomics' plus 0.005, the rounding of its 2-decimal criteria, per sample",
        "DRomics 2.6.3 drcfit (defaults, postfitfilter = FALSE)",
        largest,
        LIKELIHOOD,
        n,
    )
    rows.note(
        "DRomics: "
        + ", ".join(f"{k} {v}" for k, v in deliberate.items())
        + (f" ({'; '.join(unreported)})" if unreported else ""),
        "DRomics 2.6.3",
        f"{sum(deliberate.values())}",
        "deliberate difference",
    )
    rows.check(
        "DRomics' chosen model where both reach its maximum, off every bound and identified: |d theta| / (SE sqrt(2 max(dLL, 0.0025))), DRomics' criteria printed to 2 decimals",
        "DRomics 2.6.3 fitres",
        params,
        PARAMETER_BOUND,
        n_params,
    )
    rows.check(
        "dose of the turn of DRomics' bells and U shapes at DRomics' estimates, CytePool's closed form, relative",
        "DRomics 2.6.3 xextrem",
        turns,
        TURN_DOSE,
        n_turns,
    )
    return matched


def model_parameters_names(model: str) -> tuple[str, ...]:
    from cytepool.stats._dose_curve_models import model_of

    return model_of("dromics", model).parameters


def best_row_aicc(criteria: dict[str, Any], form: str) -> float:
    family = next(f for f, members in FAMILIES.items() if form in members)
    return float(criteria[f"InfoCrit.{family}"])


def compare_limma(rows: Rows, jobs: dict[str, Any], limma: dict[str, Any]) -> None:
    """The dose-group means, covariance and variances of the weighted and moderated
    modes, and the lack-of-fit p-values of a straight line, against limma."""
    from cytepool.stats.distributions import f_sf

    largest = {
        "means": 0.0,
        "covariance": 0.0,
        "variances": 0.0,
        "lack of fit": 0.0,
        "limma F": 0.0,
    }
    counts = {key: 0 for key in largest}
    kept_mismatch, n_jobs = 0, 0
    for name, job in jobs.items():
        if job["what"] != "fit":
            continue
        found = limma["jobs"][name]
        fits = job["fits"]
        kept_r = [job["feature_ids"][i] for i in found["kept"]]
        kept_mismatch += kept_r != job["genes"]
        n_jobs += 1
        theta = np.asarray(found["theta"], dtype=float)
        k = fits.doses.size
        to_means = np.eye(k)
        to_means[:, 0] = 1.0
        means = theta @ to_means.T
        largest["means"] = max(
            largest["means"],
            float(np.max(np.abs(means - fits.means) / np.maximum(np.abs(means), 1.0))),
        )
        counts["means"] += means.size
        for g, covariance in enumerate(found["covariance"]):
            c = to_means @ np.asarray(covariance, dtype=float) @ to_means.T
            largest["covariance"] = max(
                largest["covariance"],
                float(np.max(np.abs(c - fits.covariance[g])) / np.max(np.abs(c))),
            )
            counts["covariance"] += 1
        weighted = fits.variance == "weighted"
        s2 = np.asarray(found["sigma2"] if weighted else found["s2_post"], dtype=float)
        df = np.asarray(found["df_residual"], dtype=float) + (
            0.0 if weighted else np.asarray(found["df_prior"], dtype=float)
        )
        largest["variances"] = max(
            largest["variances"],
            float(np.max(np.abs(s2 - fits.covariance_scale) / s2)),
            float(np.max(np.abs(df - fits.df) / df)),
        )
        counts["variances"] += s2.size
        line = fits.models.index("linear")
        reference_f = np.asarray(
            found["lof_f_weighted" if weighted else "lof_f_moderated"], dtype=float
        )
        p_ref = np.asarray(f_sf(reference_f, k - 2.0, df), dtype=float)
        for g in range(p_ref.size):
            largest["lack of fit"] = max(
                largest["lack of fit"],
                p_distance(fits.lack_of_fit_p_value[g, line], p_ref[g]),
            )
        counts["lack of fit"] += p_ref.size
        if "lof_f_limma" in found and not weighted:
            p_limma = np.asarray(
                f_sf(np.asarray(found["lof_f_limma"], dtype=float), k - 2.0, df),
                dtype=float,
            )
            for g in range(p_limma.size):
                largest["limma F"] = max(
                    largest["limma F"],
                    p_distance(fits.lack_of_fit_p_value[g, line], p_limma[g]),
                )
            counts["limma F"] += p_limma.size
    rows.identical(
        "weighted and moderated: genes fitted",
        "edgeR filterByExpr",
        kept_mismatch,
        n_jobs,
    )
    rows.check(
        "dose-group means, M (m_0, theta), relative to max(|value|, 1)",
        "limma lmFit, contrasts",
        largest["means"],
        EXACT,
        counts["means"],
    )
    rows.check(
        "their unscaled covariance M V M', each gene's own under voom's weights",
        "lm.wfit, chol2inv(qr.R)",
        largest["covariance"],
        EXACT,
        counts["covariance"],
    )
    rows.check(
        "residual and moderated variances and their degrees of freedom",
        "lmFit sigma^2, eBayes s2.post, df.prior",
        largest["variances"],
        EXACT,
        counts["variances"],
    )
    rows.check(
        "lack of fit of a straight line: F on the null space of [1, x] with each gene's covariance, p relative",
        "lm.wfit, eBayes",
        largest["lack of fit"],
        EXACT,
        counts["lack of fit"],
    )
    rows.check(
        "the same with limma-trend: limma's own F of the lack-of-fit contrasts, p relative",
        "limma classifyTestsF",
        largest["limma F"],
        EXACT,
        counts["limma F"],
    )


def compare_descriptive(
    rows: Rows, fits: Any, job: dict[str, Any], limma: dict[str, Any]
) -> None:
    """A cell type without replicates at its doses: the genes, the dose-group means
    and the pooled variance of its values against R's voom E."""
    found = limma["jobs"]["descriptive"]
    kept_r = [job["feature_ids"][i] for i in found["kept"]]
    values = np.asarray(found["values"], dtype=float)  # genes x samples
    design = np.asarray(job["design"], dtype=float)
    indicators = design[:, -(fits.doses.size - 1) :]
    group = np.where(indicators.any(axis=1), indicators.argmax(axis=1) + 1, 0)
    means = np.column_stack(
        [values[:, group == k].mean(axis=1) for k in range(fits.doses.size)]
    )
    within = sum(
        ((values[:, group == k] - means[:, [k]]) ** 2).sum(axis=1)
        for k in range(fits.doses.size)
    )
    df = design.shape[0] - fits.doses.size
    rows.identical(
        "descriptive: genes fitted",
        "edgeR filterByExpr",
        int(kept_r != job["genes"]),
        1,
    )
    rows.check(
        "descriptive: dose-group means of voom's log2 CPM",
        "limma voom E",
        float(np.max(np.abs(means - fits.means) / np.maximum(np.abs(means), 1.0))),
        EXACT,
        means.size,
    )
    rows.check(
        "descriptive: pooled within-group variance",
        "limma voom E",
        float(np.max(np.abs(within / df - fits.covariance_scale) / (within / df))),
        EXACT,
        within.size,
    )
    rows.identical(
        "descriptive: lack-of-fit p-values all NaN",
        "R2",
        int(not np.isnan(fits.lack_of_fit_p_value).all()),
        fits.lack_of_fit_p_value.size,
    )


def turn_tasks(
    cases: list[Case], starts: int
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """scipy fits of the whole "dromics" set without each dose group, for the genes
    whose best fit turns."""
    tasks, turning = [], []
    for c, case in enumerate(cases):
        fits = case.fits
        if fits.model_set != "dromics":
            continue
        for g in case.genes:
            best = fits.best[g]
            if best < 0 or not np.isfinite(fits.turn_dose[g, best]):
                continue
            turning.append((c, g))
            k = fits.doses.size
            if not turns_checkable(fits.models + tuple(fits.models_not_fitted), k):
                continue
            for drop in range(k):
                keep = np.arange(k) != drop
                covariance = np.asarray(fits.covariance[g])[np.ix_(keep, keep)]
                root = np.linalg.inv(np.linalg.cholesky(covariance))
                for model in fits.models + tuple(fits.models_not_fitted):
                    tasks.append(
                        {
                            "key": f"turn|{c}|{g}|{drop}|{model}",
                            "model": documented_name("dromics", model),
                            "doses": np.asarray(fits.doses)[keep].tolist(),
                            "means": np.asarray(fits.means[g])[keep].tolist(),
                            "root": root.tolist(),
                            "starts": starts,
                            "seed": 20261231 + len(tasks),
                        }
                    )
    return tasks, turning


def group_within(case: Case, g: int) -> np.ndarray:
    """Each dose group's within-group sum of squares of a gene, from its samples."""
    assert case.samples is not None
    doses, values = case.samples.doses, case.samples.values[:, g]
    return np.array(
        [
            np.sum((values[doses == d] - values[doses == d].mean()) ** 2)
            for d in case.fits.doses
        ]
    )


def turn_dose_of(model: str, theta: np.ndarray, top: float) -> float:
    """The dose of a Gauss-probit fit's turn strictly between 0 and the top dose, from
    its documented slope, exp(-z^2/2) (-f z + (c - d) / sqrt(2 pi)), which changes
    sign at z* = (c - d) / (f sqrt(2 pi)); NaN when it does not turn there."""
    if "gauss_probit" not in model or model.endswith("monotone"):
        return np.nan
    if model.endswith("symmetric"):
        b, d, e, f = theta
        c = d
    else:
        b, c, d, e, f = theta
    if f == 0:
        return np.nan
    z = (c - d) / (f * math.sqrt(2 * math.pi))
    with np.errstate(over="ignore"):
        x = e * math.exp(min(b * z, 700.0)) if model.startswith("log") else e + b * z
    return x if 0 < x < top else np.nan


def turns_checkable(models: tuple[str, ...], k: int) -> bool:
    """Whether a turn can be checked for a group that drives it at K dose groups, by
    the documented rule: some model of the set that can turn, a Gauss-probit form
    with its f, has a spare dose group among the K - 1 groups left without one."""
    return any(
        "gauss_probit" in model
        and not model.endswith("monotone")
        and k - 1 - len(model_parameters_names(model)) >= 1
        for model in models
    )


def turn_verdict(c: int, case: Case, g: int, results: dict[str, Any]) -> float:
    """The lowest dose whose group, left out, leaves a best fit of the whole set by
    AICc that does not turn within the doses kept, by the rule coded here from the
    documentation with scipy's fits; NaN when the best fit turns without each
    group."""
    fits = case.fits
    k = fits.doses.size
    if not turns_checkable(fits.models + tuple(fits.models_not_fitted), k):
        return np.nan
    within = group_within(case, g) if fits.variance == "constant" else None
    for drop in range(k):
        keep = np.arange(k) != drop
        doses = np.asarray(fits.doses)[keep]
        n = float(np.sum(np.asarray(fits.n_samples)[keep]))
        best, best_aicc, best_theta = None, np.inf, None
        for model in (*fits.models, *fits.models_not_fitted):
            found = results.get(f"turn|{c}|{g}|{drop}|{model}")
            p = len(model_parameters_names(model))
            if found is None or found["theta"] is None or k - 1 - p < 1:
                continue
            q = found["q"]
            if within is not None:
                sum_within = float(np.sum(within[keep]))
                ll = -0.5 * n * (math.log(2 * math.pi * (sum_within + q) / n) + 1)
            else:
                s2 = float(fits.covariance_scale[g])
                cov = np.asarray(fits.covariance[g])[np.ix_(keep, keep)]
                ll = (
                    -0.5 * (k - 1) * math.log(2 * math.pi)
                    - 0.5 * ((k - 1) * math.log(s2) + np.linalg.slogdet(cov)[1])
                    - q / (2 * s2)
                )
            size = p + 1
            aicc = (
                -2 * ll
                + 2 * size
                + (
                    2 * size * (size + 1) / (n - size - 1)
                    if n - size - 1 > 0
                    else np.inf
                )
            )
            if aicc < best_aicc - 1e-8:
                best, best_aicc, best_theta = model, aicc, np.asarray(found["theta"])
        if best is None or not np.isfinite(
            turn_dose_of(best, best_theta, float(doses[-1]))
        ):
            return float(fits.doses[drop])
    return np.nan


def compare_turns(
    rows: Rows,
    cases: list[Case],
    turning: list[tuple[int, int]],
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Which dose group drives a turn, against the rule coded here with scipy's
    fits; and each turn's dose against the documented formula."""
    mismatches, verdicts = 0, []
    dose_error, n_doses = 0.0, 0
    for c, g in turning:
        case = cases[c]
        fits = case.fits
        reference = turn_verdict(c, case, g, results)
        ours = float(fits.turn_driven_by_dose[g])
        same = (np.isnan(reference) and np.isnan(ours)) or reference == ours
        mismatches += not same
        verdicts.append(
            {
                "case": case.label,
                "gene": g,
                "ours": None if np.isnan(ours) else ours,
                "reference": None if np.isnan(reference) else reference,
            }
        )
        m = fits.best[g]
        x = turn_dose_of(
            fits.models[m], model_parameters(fits, m)[g], float(fits.doses[-1])
        )
        dose_error = max(dose_error, abs(fits.turn_dose[g, m] - x) / x)
        n_doses += 1
    # Whether each best fit of the "dromics" cases turns, turning or not.
    turn_mismatches, n_best = 0, 0
    for case in cases:
        fits = case.fits
        if fits.model_set != "dromics":
            continue
        for g in case.genes:
            m = fits.best[g]
            if m < 0:
                continue
            x = turn_dose_of(
                fits.models[m], model_parameters(fits, m)[g], float(fits.doses[-1])
            )
            turn_mismatches += bool(np.isfinite(x)) != bool(
                np.isfinite(fits.turn_dose[g, m])
            )
            n_best += 1
    rows.identical(
        "whether each best fit turns strictly between 0 and the top dose",
        "the documented slope of the Gauss-probit curves",
        turn_mismatches,
        n_best,
    )
    rows.check(
        "dose of each turning best fit, relative",
        "the documented slope of the Gauss-probit curves",
        dose_error,
        EXACT,
        n_doses,
    )
    rows.identical(
        "the dose group that drives a turn: the lowest whose removal leaves a best fit of the whole set that does not turn, none at 5 dose groups",
        "the rule coded here, with scipy's fits of the set on K - 1 groups (100 starts)",
        mismatches,
        len(turning),
    )
    return verdicts


# ---------------------------------------------------------------------------------------
# Fixtures and the benchmark
# ---------------------------------------------------------------------------------------


def fixture_constant(
    case: Case,
    pybmds: list[dict[str, Any]],
    dose_r: dict[str, Any],
    scipy: dict[str, Any],
    cases: list[Case],
) -> dict[str, Any]:
    """A series' samples and the references' results on them."""
    s = case.samples
    assert s is not None
    by_key = {r["key"]: r for r in pybmds}
    out: dict[str, Any] = {
        "doses": s.doses.tolist(),
        "values": s.values.tolist(),
        "genes": s.genes,
        "shapes": s.shapes,
        "pybmds": {},
        "drc": dose_r["series"][s.name]["drc"],
        "dromics_criteria": dose_r["series"][s.name]["criteria"],
        "dromics_fitres": dose_r["series"][s.name]["fitres"],
        "scipy": {},
    }
    for key, found in by_key.items():
        name, model, g, increasing = key.split("|")
        if name == s.name:
            out["pybmds"].setdefault(model, {}).setdefault(g, {})[increasing] = {
                k: v for k, v in found.items() if k != "key"
            }
    for key, found in scipy.items():
        if key.startswith("turn"):
            continue
        c, m, g = map(int, key.split("|"))
        other = cases[c]
        if other.samples is not None and other.samples.name == s.name:
            out["scipy"].setdefault(other.fits.model_set, {}).setdefault(
                other.fits.models[m], {}
            )[str(g)] = found["q"]
    return out


def fixture_plugin(
    cases: list[Case], scipy: dict[str, Any], prefix: str
) -> dict[str, Any]:
    """The means, covariance and variances of a cell type's weighted or moderated
    fits, and scipy's optimum of every set's models on them."""
    chosen = [
        (c, case)
        for c, case in enumerate(cases)
        if case.label.startswith(prefix) and case.genes
    ]
    first = chosen[0][1]
    genes = first.genes
    out: dict[str, Any] = {
        "genes": genes,
        "doses": np.asarray(first.fits.doses).tolist(),
        "n_samples": np.asarray(first.fits.n_samples).tolist(),
        "means": np.asarray(first.fits.means)[genes].tolist(),
        "covariance": np.asarray(first.fits.covariance)[genes].tolist(),
        "covariance_scale": np.asarray(first.fits.covariance_scale)[genes].tolist(),
        "df": np.asarray(first.fits.df)[genes].tolist(),
        "variance": first.fits.variance,
        "scipy": {},
    }
    for c, case in chosen:
        for key, found in scipy.items():
            if key.startswith("turn"):
                continue
            kc, m, g = map(int, key.split("|"))
            if kc == c:
                out["scipy"].setdefault(case.fits.model_set, {}).setdefault(
                    case.fits.models[m], {}
                )[str(genes.index(g))] = found["q"]
    return out


def turn_case_record(case: Case) -> dict[str, Any]:
    """What the tests need to fit a case of constructed turns again: its samples under
    constant variance, or its means, covariance and variance."""
    if case.samples is not None:
        return {
            "doses": case.samples.doses.tolist(),
            "values": case.samples.values.tolist(),
        }
    fits = case.fits
    return {
        "variance": fits.variance,
        "doses": np.asarray(fits.doses).tolist(),
        "n_samples": np.asarray(fits.n_samples).tolist(),
        "means": np.asarray(fits.means).tolist(),
        "covariance": np.asarray(fits.covariance).tolist(),
        "covariance_scale": np.asarray(fits.covariance_scale).tolist(),
        "df": np.asarray(fits.df).tolist(),
    }


def series_record(name: str) -> dict[str, Any]:
    """A simulated series as the fixtures keep it: its samples, counts and cells per
    cell type and its formula, so that the tests rebuild the cube and design without
    the simulator."""
    cube, _, design = simulate(name)
    samples = design.samples
    record: dict[str, Any] = {
        "formula": design.formula,
        "sample_id": samples["sample_id"].tolist(),
        "dose": np.asarray(samples["dose"], dtype=float).tolist(),
        "groups": cube.groups["group_id"].tolist(),
        "features": cube.features["feature_id"].tolist(),
        "counts": [
            np.asarray(cube.layer("counts", g)).astype(int).tolist()
            for g in range(len(cube.groups["group_id"]))
        ],
        "n_cells": np.asarray(cube.n_cells).astype(int).tolist(),
    }
    if "sex" in samples:
        record["sex"] = samples["sex"].tolist()
    return record


def write_fixtures(fixtures: dict[str, Any]) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, content in fixtures.items():
        path = FIXTURE_DIRECTORY / name
        path.write_text(
            json.dumps(content, separators=(",", ":"), allow_nan=True) + "\n"
        )
        print(
            f"wrote {path.relative_to(FIXTURE_DIRECTORY.parents[2])} ({path.stat().st_size / 1024:.0f} KB)"
        )


def benchmark() -> None:
    """fit_dose_curves with "epa" and "dromics" on 10,000 genes x 9 dose groups x 3,
    and dose_response's weighted curves on one cell type of 10,000 genes x 27
    profiles, against the ordinary least squares fit of 20,000 genes x 200 samples of
    linear_models.py's benchmark, the machine's baseline."""
    from cytepool.dose_responses import dose_response
    from cytepool.design import Design
    from cytepool.simulated_dose_series import simulate_dose_series
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.linear_models import fit_linear_model

    cube, _ = simulate_dose_series(
        HALF_LOG, replicates=3, cell_types=1, cells=300.0, n_genes=10_000, rng=20261225
    )
    design = Design(cube.samples, "~ 1")
    rng = np.random.default_rng(1)
    doses = np.repeat(HALF_LOG, 3)
    values = 6.0 + rng.normal(0.0, 0.3, (doses.size, 10_000))
    y = rng.normal(size=(200, 20_000))
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    runs = {
        "fit_dose_curves, epa": lambda: fit_dose_curves(doses, values, models="epa"),
        "fit_dose_curves, dromics": lambda: fit_dose_curves(
            doses, values, models="dromics"
        ),
        "dose_response, epa, weighted": lambda: dose_response(
            cube, design, dose=axis(), models="epa", variance="weighted"
        ),
        "dose_response without curves": lambda: dose_response(
            cube, design, dose=axis()
        ),
        "fit_linear_model, 20,000 genes x 200 samples (baseline)": lambda: (
            fit_linear_model(y, x)
        ),
    }
    timings = {}
    for name, run in runs.items():
        run()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        timings[name] = float(np.median(times))
    baseline = timings["fit_linear_model, 20,000 genes x 200 samples (baseline)"]
    print("| run | seconds | baselines |")
    print("| --- | --- | --- |")
    for name, seconds in timings.items():
        print(f"| {name} | {seconds:.2f} | {seconds / baseline:.0f} |")
    print("target: 60 s per cell type of 10,000 genes, about 350 baselines")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--benchmark", action="store_true", help="time the fits")
    parser.add_argument("--scipy", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pybmds", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.scipy is not None:
        scipy_main(arguments.scipy)
        return 0
    if arguments.pybmds is not None:
        pybmds_main(arguments.pybmds)
        return 0
    if arguments.benchmark:
        benchmark()
        return 0
    start = time.perf_counter()
    found = constant_samples()
    cases = constant_cases(found + [shifted(found[0])])
    plugin, jobs = plugin_cases(PLUGIN_GENES)
    descriptive_fits, descriptive = descriptive_job()
    jobs["descriptive"] = descriptive
    constructed = constructed_turn_cases(plugin)
    all_cases = cases + plugin
    print(f"CytePool fitted in {time.perf_counter() - start:.0f} s")

    pybmds_inputs = []
    for case in cases:
        s = case.samples
        if case.fits.model_set != "epa" or s is None:
            continue
        for model in PYBMDS_MODELS:
            if model not in case.fits.models:
                continue
            for g in case.genes:
                for increasing in (True, False):
                    pybmds_inputs.append(
                        {
                            "key": f"{s.name}|{model}|{g}|{increasing}",
                            "model": model,
                            "doses": s.doses.tolist(),
                            "values": s.values[:, g].tolist(),
                            "increasing": increasing,
                        }
                    )
    dose_inputs = {
        "series": {
            s.name: {"doses": s.doses.tolist(), "values": s.values.T.tolist()}
            for s in found
        }
    }
    tasks = scipy_tasks(
        [
            c
            for c in all_cases
            if not (c.samples is not None and c.samples.name.endswith("+500"))
        ],
        STARTS,
    )
    # The keys of scipy's tasks count the cases in all_cases' order without the shifted
    # series, so map them back.
    kept_cases = [
        c
        for c in all_cases
        if not (c.samples is not None and c.samples.name.endswith("+500"))
    ]
    turn_cases = kept_cases + constructed
    turns, turning = turn_tasks(turn_cases, TURN_STARTS)
    limma_payload = {
        "jobs": {
            name: {k: v for k, v in job.items() if k != "fits"}
            for name, job in jobs.items()
        }
    }

    start = time.perf_counter()
    bmds = run_python(
        "--pybmds",
        os.environ.get("BMDS_PYTHON", "python"),
        {"pybmds_inputs.json": json.dumps(pybmds_inputs)},
        "pybmds.json",
    )
    dose_r = run_r(DOSE_R, dose_inputs, "DOSE_RSCRIPT")
    limma = run_r(LIMMA_R, limma_payload, "RSCRIPT")
    scipy = run_python(
        "--scipy",
        sys.executable,
        {"scipy_inputs.json": json.dumps(tasks + turns)},
        "scipy.json",
    )
    print(f"References ran in {time.perf_counter() - start:.0f} s")
    versions = (
        bmds["versions"] | dose_r["versions"] | limma["versions"] | scipy["versions"]
    )
    print("Reference: " + ", ".join(f"{k} {v}" for k, v in versions.items()))

    rows = Rows()
    results = {r["key"]: r for r in scipy["results"]}
    compare_scipy(
        rows, kept_cases, {k: v for k, v in results.items() if not k.startswith("turn")}
    )
    compare_pybmds(rows, cases, bmds["results"])
    compare_lack_of_fit(rows, cases, dose_r)
    compare_drc(rows, cases, dose_r)
    matched = compare_dromics(rows, cases, dose_r)
    compare_limma(rows, jobs, limma)
    compare_descriptive(rows, descriptive_fits, descriptive, limma)
    verdicts = compare_turns(rows, turn_cases, turning, results)
    not_converged = sum(
        int(np.count_nonzero(~c.fits.converged & (c.fits.not_fitted_reason == "")))
        for c in all_cases
    )
    fitted = sum(
        int(np.count_nonzero(c.fits.not_fitted_reason == "")) for c in all_cases
    )
    rows.note(
        f"fits that stopped at 500 iterations before converging: {not_converged} of {fitted}",
        "CytePool",
        f"{not_converged}",
        "information",
    )
    print()
    rows.print()
    if arguments.write_fixtures:
        if rows.failures:
            print("not writing fixtures: some checks failed")
        else:
            by_name = {
                c.samples.name: c
                for c in cases
                if c.samples is not None and c.fits.model_set == "epa"
            }
            references = {
                "versions": versions,
                "constant": {
                    name: fixture_constant(
                        by_name[name], bmds["results"], dose_r, results, kept_cases
                    )
                    for name in ("ladder", "decades", "four")
                },
                "weighted": fixture_plugin(
                    kept_cases, results, "ladder abundant weighted"
                ),
                "moderated": fixture_plugin(
                    kept_cases, results, "ladder abundant moderated"
                ),
                "turns": verdicts,
                "turn_cases": {
                    case.label: turn_case_record(case) for case in constructed
                },
                "dromics_matched": [
                    {k: v for k, v in item.items() if k != "row"} | {"row": item["row"]}
                    for item in matched[:12]
                ],
            }
            limma_fixture = {
                "versions": limma["versions"],
                "series": {
                    name: series_record(name)
                    for name in ("ladder", "sex", "descriptive")
                },
                "jobs": {
                    name: {
                        k: v
                        for k, v in limma["jobs"][name].items()
                        if k
                        in (
                            "kept",
                            "sigma2",
                            "df_residual",
                            "s2_post",
                            "df_prior",
                            "lof_f_weighted",
                            "lof_f_moderated",
                            "lof_f_limma",
                            "values",
                        )
                    }
                    | (
                        {
                            "theta": limma["jobs"][name]["theta"][:20],
                            "covariance": limma["jobs"][name]["covariance"][:20],
                        }
                        if "theta" in limma["jobs"][name]
                        else {}
                    )
                    for name in (
                        "ladder abundant voom",
                        "ladder abundant trend",
                        "sex cell_type_1 voom",
                        "sex cell_type_1 trend",
                        "descriptive",
                    )
                },
            }
            write_fixtures({"references.json": references, "limma.json": limma_fixture})
    print(f"{len(rows.rows)} checks, {rows.failures} failed")
    return 1 if rows.failures else 0


if __name__ == "__main__":
    sys.exit(main())
