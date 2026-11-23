"""Fit dose-response curves of named models to every gene, and compare them.

A moderated test says that a gene responds to a dose, not what shape the response
takes. :func:`fit_dose_curves` fits a named set of models to each gene of a samples x
genes matrix of log-scale values, such as log2 CPM, and :class:`DoseCurveFits` holds
each fit with its parameters, log-likelihood, AIC and AICc (Akaike's criterion
corrected for small samples), Akaike weight (each model's share of support among the
gene's models), lack-of-fit test, largest scaled residual and the gene's best model.
:func:`~cytepool.dose_response` fits the same sets to the dose groups of each cell type,
with the variances of its moderated fit. The sets, their means and bounds, the
variance modes and how to read the fits are in
https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-model-curves.

The sets:
    ``"epa"``: the eight models of NTP 2018 and of EPA's Transcriptomic Assessment
    Products (US EPA 2024), linear, polynomial of degree 2, power, Hill and exponential
    2 to 5, in the forms and bounds of BMDS (US EPA 2026), with BMDS's parameter
    letters; ``"ll4"``: the four-parameter log-logistic curve of drc's ``LL.4`` (Ritz
    et al. 2015); ``"dromics"``: DRomics' families (Larras et al. 2018;
    Delignette-Muller et al. 2023), linear, exponential, the log-logistic curve and
    the Gauss-probit and log-Gauss-probit curves, which can rise and then fall, each
    of their four- and five-parameter forms a model of its own. Hill, sigmoid Emax
    and four-parameter log-logistic curves are one family (Kappenberg et al. 2023):
    ``"ll4"``'s curve is the Hill curve with a slope from 0.1.

The fits:
    Each gene's dose-group means are sufficient for its fit: under constant variance
    the samples' sum of squares about a curve is the within-group sum of squares
    plus Q = sum_k n_k (ybar_k - f(x_k))^2, so each model is fitted to the K means
    with weights n_k. Each fit starts from the three lowest local minima of a fixed
    grid of its nonlinear parameters, at which the linear ones are solved exactly
    (variable projection; Golub and Pereyra 1973), and each start is refined within
    the model's bounds, so that no fit needs a seed or a start and the larger maximum
    is found where a search from one start can stop short.

Model choice:
    AIC = -2 LL + 2k and AICc = AIC + 2k(k + 1) / (n - k - 1) count k = p + 1
    parameters, the p of the mean and the variance, and n samples (Hurvich and Tsai
    1989); AICc is infinite when n <= k + 1. Akaike weights are exp(-Delta / 2) over
    their sum, Delta the AICc minus the gene's lowest (Burnham and Anderson 2002).

References:
    US EPA (2026). BMDS User Guide: continuous endpoints. Online documentation of
    BMDS Online, BMDS Desktop and pybmds 25.2.
    https://usepa.github.io/bmds-user-guide/continuous.html

    US EPA (2024). Standard Methods for Development of EPA Transcriptomic Assessment
    Products (ETAPs). EPA/600/X-23/083, US Environmental Protection Agency, Office of
    Research and Development. doi:10.23645/epacomptox.25365496

    NTP (2018). NTP research report on National Toxicology Program approach to
    genomic dose-response modeling. NTP Research Report 5. National Institute of
    Environmental Health Sciences. doi:10.22427/ntp-rr-5

    Ritz C, Baty F, Streibig JC, Gerhard D (2015). Dose-response analysis using R.
    PLOS ONE 10, e0146021. doi:10.1371/journal.pone.0146021

    Larras F, Billoir E, Baillard V, Siberchicot A, Scholz S, Wubet T, Tarkka M,
    Schmitt-Jansen M, Delignette-Muller ML (2018). DRomics: a turnkey tool to support
    the use of the dose-response framework for omics data in ecological risk
    assessment. Environmental Science & Technology 52, 14461-14468.
    doi:10.1021/acs.est.8b04752

    Delignette-Muller ML, Siberchicot A, Larras F, Billoir E (2023). DRomics, a
    workflow to exploit dose-response omics data in ecotoxicology. Peer Community
    Journal 3, e90. doi:10.24072/pcjournal.325

    Kappenberg F, Duda JC, Schürmeyer L, Gül O, Brecklinghaus T, Hengstler JG,
    Schorning K, Rahnenführer J (2023). Guidance for statistical design and analysis
    of toxicological dose-response experiments, based on a comprehensive literature
    review. Archives of Toxicology 97, 2741-2761. doi:10.1007/s00204-023-03561-w

    Golub GH, Pereyra V (1973). The differentiation of pseudo-inverses and nonlinear
    least squares problems whose variables separate. SIAM Journal on Numerical
    Analysis 10, 413-432. doi:10.1137/0710036

    Hurvich CM, Tsai CL (1989). Regression and time series model selection in small
    samples. Biometrika 76, 297-307. doi:10.1093/biomet/76.2.297

    Burnham KP, Anderson DR (2002). Model Selection and Multimodel Inference, 2nd
    edition. Springer, New York. doi:10.1007/b97636

    Pinheiro J, Bornkamp B, Glimm E, Bretz F (2014). Model-based dose finding under
    model uncertainty using general parametric models. Statistics in Medicine 33,
    1646-1661. doi:10.1002/sim.6052

    US EPA (2012). Benchmark Dose Technical Guidance. EPA/100/R-12/001, Risk
    Assessment Forum, US Environmental Protection Agency, Washington, DC.

    Ringblom J, Johanson G, Öberg M (2014). Current modeling practice may lead to
    falsely high benchmark dose estimates. Regulatory Toxicology and Pharmacology 69,
    171-177. doi:10.1016/j.yrtph.2014.03.004

    EFSA Scientific Committee (2021). Opinion on the impact of non-monotonic dose
    responses on EFSA's human health risk assessments. EFSA Journal 19, e06877.
    doi:10.2903/j.efsa.2021.6877
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._columns import text_column
from cytepool._doses import checked_doses, distinct_doses
from cytepool._jit import by_chunks, thread_limit
from cytepool._masked_arrays import read_only, real_array, reject_masked, unmasked_array
from cytepool._messages import close_matches, fitted_lines, quoted
from cytepool.stats._dose_curve_models import (
    DROMICS_EXPONENTIAL,
    LINE,
    POLYNOMIAL,
    POSITIVE_MEANS,
    QUADRATIC,
    SET_PARAMETERS,
    SETS,
    TURNING,
    DoseScale,
    Model,
    bounds,
    constraint,
    exponential_sign_gap,
    grid,
    model_of,
    n_linear,
    reported,
    turn_doses,
)
from cytepool.stats._dose_curve_search import (
    fit_genes_parallel,
    fit_genes_serial,
    grid_bases,
    whiten_bases,
    whitening_roots,
)
from cytepool.stats._result_tables import feature_id_column
from cytepool.stats.distributions import f_sf
from cytepool.table import Table

__all__ = ["DoseCurveFits", "fit_dose_curves"]

FloatArray = npt.NDArray[np.float64]

MODEL_SETS = tuple(SETS)
"""The model sets, ``"epa"``, ``"bmdexpress"``, ``"ll4"`` and ``"dromics"``."""
VARIANCES = ("constant", "weighted", "moderated")
"""How the curves weigh the dose-group means: as samples of constant variance, or
by the covariance of the means of a moderated fit with its residual or moderated
variance."""
CONSTANT = "constant"
TIE = 1e-8
"""AICc within this of the lowest counts as a tie, which the first model of the set
wins."""
LARGEST_VALUE = 1e100
"""The largest size of a value: far beyond log-scale values such as log2 CPM, and far
enough below the largest double that the fits' sums of squares stay finite."""


@dataclass(frozen=True, eq=False, repr=False)
class DoseCurveFits:
    """The fits of a set of dose-response models to every gene, with the data they
    were fitted to.

    Rows of the arrays are genes, in the order given; the model axis holds the
    set's models fitted, in set order (``models``). The arrays are read-only.
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-model-curves
    gives each model's mean and bounds and explains how to read the fits.

    Attributes:
        model_set: ``"epa"``, ``"bmdexpress"``, ``"ll4"`` or ``"dromics"``.
        models: The set's models fitted, in set order.
        models_not_fitted: The set's other models, each with the reason: a model of
            p parameters needs a spare dose group, K - p of at least 1.
        parameter_names: The set's parameter letters, the last axis of
            ``parameters``: BMDS's for ``"epa"`` and ``"bmdexpress"``, where Hill's k
            is the EC50, the dose of half the change v; DRomics' for the others,
            where the
            log-logistic curve ``ll4`` has b > 0 its slope, d its level at dose 0, c
            its level at high doses and e its EC50, the dose of half the change.
            drc's ``LL.4`` estimates (b, c, d, e) with b < 0 are (-b, d, c, e).
        variance: How the means were weighed: ``"constant"``, ``"weighted"`` or
            ``"moderated"``.
        doses: ``(K,)`` the dose of each dose group, ascending.
        n_samples: ``(K,)`` the samples of each dose group.
        means: ``(n_genes, K)`` the dose-group means the curves were fitted to.
        covariance: ``(n_genes, K, K)`` their unscaled covariance, diag(1 / n_k)
            under constant variance.
        covariance_scale: ``(n_genes,)`` the variance that multiplies
            ``covariance`` to give the covariance of the means: the pooled
            within-group variance under constant variance, NaN without replicates;
            the residual variance of the cell type's moderated fit in
            :func:`~cytepool.dose_response` (``"weighted"``); or its moderated
            variance (``"moderated"``).
        df: ``(n_genes,)`` the degrees of freedom of ``covariance_scale``.
        within_group_sd: ``(n_genes, K)`` under constant variance, the sample SD of
            each dose group's values, with n_k - 1 in the denominator; NaN for a
            group of one sample, and in the weighted and moderated modes, whose
            means come from a fit rather than from samples.
        parameters: ``(n_genes, n_models, n_parameter_names)`` the parameters of
            each fit, NaN where a model has no such parameter or was not fitted.
        log_likelihood: ``(n_genes, n_models)``: under constant variance that of the
            samples with the variance at its maximum, -n/2 (log(2 pi sigma^2) + 1)
            with sigma^2 the residual sum of squares over n, as BMDS and drc report
            it; otherwise that of the means given their covariance.
        aic: ``(n_genes, n_models)`` AIC, counting every parameter of the mean and
            the variance, also those at a bound.
        aicc: ``(n_genes, n_models)`` AICc, AIC corrected for small samples, with the
            same count.
        akaike_weight: ``(n_genes, n_models)`` each fitted model's share of the
            gene's support, exp(-Delta / 2) over the sum, Delta its AICc minus the
            lowest; 0 for an infinite AICc, NaN where not fitted.
        lack_of_fit_p_value: ``(n_genes, n_models)`` the F-test of the curve against
            free dose-group means, F = Q / ((K - p) s^2) on K - p and ``df``
            degrees of freedom, s^2 the ``covariance_scale`` and Q the weighted sum
            of squares of the means about the curve; read at 0.1 (US EPA 2012). It
            is exact for curves linear in their parameters; under constant
            variance it is the pure-error test of lack of fit, of which BMDS's test 4,
            n log(1 + (K - p) F / df) on a chi-square of K - p degrees of freedom, is
            the large-sample form. NaN without replicates and for cell types that
            support only description.
        max_abs_scaled_residual: ``(n_genes, n_models)`` the largest
            |ybar_k - f(x_k)| / se_k over the doses, se_k = sigma / sqrt(n_k) under
            constant variance and s sqrt(C_kk) otherwise; above 2 warrants a look.
        parameters_at_bounds: ``(n_genes, n_models)`` the parameters of each fit
            that sit at a bound, comma-separated, empty for none.
        turn_dose: ``(n_genes, n_models)`` the dose at which the fit's slope changes
            sign strictly between 0 and the top dose; NaN when it does not turn
            there, as a curve that peaks at the top dose does not. Only Gauss-probit
            curves with f other than 0 can turn.
        converged: ``(n_genes, n_models)`` False when the refinement stopped at its
            limit of 500 iterations before converging; the fit keeps the best point
            found and its statistics.
        not_fitted_reason: ``(n_genes, n_models)`` why a model was not fitted to the
            gene, empty when it was: exponential 2 to 5 where a dose-group mean is
            not positive, and every model when the values do not vary or, in the
            weighted and moderated modes, when the means' variance is 0 to rounding.
        best: ``(n_genes,)`` the position in ``models`` of the gene's lowest AICc,
            the first in set order within 1e-8 of it; -1 when no AICc is finite.
        turn_driven_by_dose: ``(n_genes,)`` for a gene whose best fit turns, the
            lowest dose whose group, left out, leaves a best fit of the set that does
            not turn within the doses kept; 0 when that group is the vehicle's, so
            test it with ``numpy.isfinite``. NaN when the best fit turns without each
            group, for genes whose best fit does not turn, and for every gene when
            no model that can turn has a spare dose group among the K - 1 groups
            left, as at 5 dose groups; the printout then says that the turns were
            not checked.
    """

    model_set: str
    models: tuple[str, ...]
    models_not_fitted: Mapping[str, str]
    parameter_names: tuple[str, ...]
    variance: str
    doses: FloatArray
    n_samples: FloatArray
    means: FloatArray
    covariance: FloatArray
    covariance_scale: FloatArray
    df: FloatArray
    within_group_sd: FloatArray
    parameters: FloatArray
    log_likelihood: FloatArray
    aic: FloatArray
    aicc: FloatArray
    akaike_weight: FloatArray
    lack_of_fit_p_value: FloatArray
    max_abs_scaled_residual: FloatArray
    parameters_at_bounds: np.ndarray
    turn_dose: FloatArray
    converged: np.ndarray
    not_fitted_reason: np.ndarray
    best: np.ndarray
    turn_driven_by_dose: FloatArray

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values["models_not_fitted"] = dict(self.models_not_fitted)
        return (_frozen_fits, (values,))

    def table(self, *, feature_ids: Any = None) -> Table:
        """Return the fits as a table, one row per gene and model fitted.

        Genes in order, and each gene's models in set order: ``feature_id``,
        ``model``, the set's parameter columns (NaN where the model has no such
        parameter), ``n_parameters``, ``spare_groups`` (K - p), ``log_likelihood``,
        ``aic``, ``aicc``, ``akaike_weight``, ``best`` (True for the gene's best
        model), ``lack_of_fit_p_value``, ``max_abs_scaled_residual``,
        ``parameters_at_bounds``, ``turn_dose``, ``turn_driven_by_dose`` (on the best
        row; NaN on the others), ``converged`` and ``not_fitted_reason``.

        Args:
            feature_ids: One id per gene, as text; by default ``feature_0``,
                ``feature_1``, ...

        Raises:
            ValueError: If ``feature_ids`` does not give one id per gene.
            TypeError: If an id is not text, or ``feature_ids`` is a numpy masked
                array.
        """
        n_genes, n_models = self.log_likelihood.shape
        ids = feature_id_column(feature_ids, n_genes)
        genes = np.repeat(np.arange(n_genes), n_models)
        model_of_row = np.tile(np.arange(n_models), n_genes)
        sizes = np.array(
            [model_of(self.model_set, name).n_parameters for name in self.models],
            dtype=np.int64,
        )
        best = self.best[genes] == model_of_row
        driven = np.where(best, self.turn_driven_by_dose[genes], np.nan)
        columns: dict[str, Any] = {
            "feature_id": ids[genes],
            "model": text_column([self.models[m] for m in model_of_row]),
        }
        for j, name in enumerate(self.parameter_names):
            columns[name] = self.parameters[:, :, j].ravel()
        columns |= {
            "n_parameters": sizes[model_of_row],
            "spare_groups": self.doses.size - sizes[model_of_row],
            "log_likelihood": self.log_likelihood.ravel(),
            "aic": self.aic.ravel(),
            "aicc": self.aicc.ravel(),
            "akaike_weight": self.akaike_weight.ravel(),
            "best": best,
            "lack_of_fit_p_value": self.lack_of_fit_p_value.ravel(),
            "max_abs_scaled_residual": self.max_abs_scaled_residual.ravel(),
            "parameters_at_bounds": text_column(
                self.parameters_at_bounds.ravel().tolist()
            ),
            "turn_dose": self.turn_dose.ravel(),
            "turn_driven_by_dose": driven,
            "converged": self.converged.ravel(),
            "not_fitted_reason": text_column(self.not_fitted_reason.ravel().tolist()),
        }
        return Table(columns)

    def __repr__(self) -> str:
        n_genes = self.means.shape[0]
        lines = [
            f"DoseCurveFits ({self.model_set}, {self.variance} variance): "
            f"{n_genes:,} gene{'' if n_genes == 1 else 's'}, {self.doses.size} dose "
            f"groups; models: {', '.join(self.models)}"
        ]
        if self.models_not_fitted:
            lines.append(
                "  not fitted (no spare dose group): "
                + ", ".join(self.models_not_fitted)
            )
        lines.append("  best: " + _best_counts(self))
        turns = _turn_text(self)
        if turns:
            lines.append("  " + turns)
        return fitted_lines("\n".join(lines))


def _frozen_fits(values: dict[str, Any]) -> DoseCurveFits:
    """DoseCurveFits of read-only arrays and an immutable mapping of the models not
    fitted, as _fit_means builds them; pickled fits are rebuilt by it."""
    frozen: dict[str, Any] = {
        name: read_only(value) if isinstance(value, np.ndarray) else value
        for name, value in values.items()
    }
    frozen["models_not_fitted"] = MappingProxyType(dict(values["models_not_fitted"]))
    return DoseCurveFits(**frozen)


def _best_counts(fits: DoseCurveFits) -> str:
    """The genes whose best model each model is, most first, for printouts."""
    counts = np.bincount(fits.best[fits.best >= 0], minlength=len(fits.models))
    order = sorted(range(len(fits.models)), key=lambda m: (-counts[m], m))
    parts = [f"{fits.models[m]} {counts[m]:,}" for m in order if counts[m]]
    none = int(np.count_nonzero(fits.best < 0))
    if none:
        parts.append(f"none {none:,}")
    return ", ".join(parts) or "none"


def _turning(best: np.ndarray, turn_dose: FloatArray) -> np.ndarray:
    """Whether each gene's best fit turns within the doses."""
    rows = np.arange(best.size)
    turns = np.isfinite(turn_dose[rows, np.maximum(best, 0)]) & (best >= 0)
    return np.asarray(turns, dtype=bool)


def _unchecked_turns(model_set: str, n_groups: int) -> str:
    """Why no turn can be checked for a dose group that drives it at K dose groups,
    empty when turns can be checked or no model of the set can turn: no model that
    can turn has a spare dose group among the K - 1 groups left without one."""
    turning = [m.n_parameters for m in SETS[model_set] if m.kind in TURNING]
    if not turning or n_groups - 1 - min(turning) >= 1:
        return ""
    return (
        f"no model that can turn has a spare dose group among the {n_groups - 1} "
        f"left without one of the {n_groups} dose groups: the smallest has "
        f"{min(turning)} parameters"
    )


def _turn_text(fits: DoseCurveFits) -> str:
    """The best fits that turn and those that one dose group drives, for printouts;
    empty when no model fitted can turn."""
    if not any(model_of(fits.model_set, m).kind in TURNING for m in fits.models):
        return ""
    turning = int(np.count_nonzero(_turning(fits.best, fits.turn_dose)))
    text = f"{turning:,} best fit{' turns' if turning == 1 else 's turn'}"
    if _unchecked_turns(fits.model_set, fits.doses.size):
        return (
            f"{text}, not checked for a dose group that drives them: no model that "
            f"can turn fits {fits.doses.size - 1} dose groups"
        )
    driven = int(np.count_nonzero(np.isfinite(fits.turn_driven_by_dose)))
    return f"{text}, {driven:,} driven by one dose group"


def _checked_model_set(models: Any) -> str:
    """The name of a model set; a model's name, or a list of names, is refused with
    the sets that hold it."""
    reject_masked(models, "models")
    if isinstance(models, str) and models in SETS:
        return models
    names = [models] if isinstance(models, str) else models
    homes = []
    if isinstance(names, list | tuple):
        for name in names:
            holding = [
                key
                for key, members in SETS.items()
                if isinstance(name, str) and any(m.name == name for m in members)
            ]
            if holding:
                homes.append(f"{name!r} is in {quoted(holding)}")
    message = (
        "models names a set of models, 'epa' (linear, polynomial_2, power, hill and "
        "exponential_2 to exponential_5), 'll4' (the log-logistic curve), "
        "'bmdexpress' (the models of 'epa' with the bounds of BMDS 2's programs, as "
        "BMDExpress fits them) or 'dromics' (linear, exponential, ll4 and the "
        f"Gauss-probit curves), got {models!r}"
    )
    if homes:
        message += "; " + ", ".join(homes) + ": pass the set"
    elif isinstance(models, str):
        message += close_matches(models.lower(), MODEL_SETS)
    error = ValueError if isinstance(models, str) else TypeError
    raise error(message)


def fit_dose_curves(
    doses: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    models: str = "epa",
    n_threads: int | None = None,
) -> DoseCurveFits:
    """Fit a set of dose-response models to every gene, under constant variance.

    Each gene's values are fitted against dose by maximum likelihood with a constant
    variance, as BMDS, drc and DRomics fit individual data: through the dose-group
    means with weights n_k, which give the samples' likelihood. Each model is fitted
    from the lowest points of a fixed grid of its nonlinear parameters and refined
    within its bounds, so the results need no seed and do not depend on the number
    of threads. A model of p parameters is fitted only with a spare dose group, K -
    p of at least 1 (Ringblom et al. 2014); exponential 2 to 5 are not fitted to a
    gene with a dose-group mean at or below 0, and no model to a gene whose values
    do not vary. A gene whose best fit turns is fitted again without each dose
    group in turn, to show whether one group drives the turn (EFSA 2021).
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-model-curves
    gives the models' means and bounds and how to read the fits.

    Args:
        doses: One dose per sample, finite and at least 0; the dose groups are the
            distinct doses. No vehicle (dose 0) is required.
        values: ``(n_samples, n_genes)`` log-scale values, such as log2 CPM, finite
            and at most 1e100 in size.
        models: The set of models: ``"epa"`` (default; linear, polynomial_2, power,
            hill, where k is the EC50, the dose of half the change v, and
            exponential_2 to exponential_5, as BMDS writes them), ``"ll4"`` (the
            four-parameter log-logistic curve: b > 0 its slope, d its level at dose
            0, c its level at high doses and e its EC50, the dose of half the
            change; drc's ``LL.4`` estimates (b, c, d, e) with b < 0 are (-b, d, c,
            e)), ``"bmdexpress"`` (the models of ``"epa"`` with the bounds of BMDS
            2's programs, as BMDExpress fits them: an unrestricted quadratic, and
            Hill's k and exponential 4 and 5's rate bounded only from below) or
            ``"dromics"`` (DRomics' linear, exponential and log-logistic curves and
            its Gauss-probit forms).
        n_threads: The most threads to use; None keeps numba's setting.

    Returns:
        The fits, with the dose-group means and covariance they were fitted to.

    Raises:
        TypeError: If ``doses`` or ``values`` do not hold real numbers, are numpy
            masked arrays, or ``models`` is not a string.
        ValueError: If ``values`` is not 2-D with one row per dose, a value or dose
            is not finite, a value is larger than 1e100 in size, a dose is
            negative, two distinct doses differ by less than 1e-8 of the dose, the
            doses form fewer than two groups, ``models`` names no set, or no model
            of the set has a spare dose group.

    Example:
        >>> import numpy as np
        >>> doses = np.repeat([0.0, 0.1, 0.3, 1.0, 3.0, 10.0], 3)
        >>> rng = np.random.default_rng(1)
        >>> mean = 6.0 + 2.0 / (1.0 + (1.0 / np.maximum(doses, 1e-12)) ** 2)
        >>> values = (mean + rng.normal(0.0, 0.1, doses.size))[:, None]
        >>> fits = fit_dose_curves(doses, values, models="ll4")
        >>> [round(float(v), 1) for v in fits.parameters[0, 0]]  # b, c, d, e
        [2.1, 8.0, 6.0, 1.0]
    """
    reject_masked(doses, "doses")
    reject_masked(values, "values")
    x = checked_doses(doses, "doses")
    y = real_array(unmasked_array(values, "values"), "values", boolean=False)
    if y.ndim != 2:
        raise ValueError(
            "values must be a 2-D array of shape (n_samples, n_genes), got shape "
            f"{y.shape}"
        )
    if y.shape[0] != x.size:
        raise ValueError(
            f"values has {y.shape[0]} rows and doses {x.size} entries; rows of values "
            "are samples, one dose each"
        )
    if not np.isfinite(y).all():
        raise ValueError("values must be finite; leave out genes with missing values")
    largest = float(np.abs(y).max(initial=0.0))
    if largest > LARGEST_VALUE:
        raise ValueError(
            f"values must be at most {LARGEST_VALUE:g} in size, as log-scale values "
            f"such as log2 CPM are, got {largest:g}; the sums of squares of larger "
            "values overflow"
        )
    model_set = _checked_model_set(models)
    levels = distinct_doses(x, "(the doses' unit)")
    if levels.size < 2:
        raise ValueError(
            f"a dose-response curve needs two dose groups or more, and every sample has "
            f"the dose {levels[0]:g}"
        )
    reason = _no_model_fits(model_set, levels.size)
    if reason:
        raise ValueError(reason)
    return _fit_means(_constant_variance_data(x, y), model_set, n_threads=n_threads)


def _no_model_fits(model_set: str, n_groups: int) -> str:
    """Why no model of a set can be fitted at K dose groups, empty when one can."""
    smallest = min(m.n_parameters for m in SETS[model_set])
    if n_groups > smallest:
        return ""
    return (
        f"no model of the set {model_set!r} has a spare dose group at {n_groups} "
        f"dose groups: its smallest model has {smallest} parameters and needs "
        f"{smallest + 1} dose groups"
    )


# ---------------------------------------------------------------------------------------
# The data of a fit
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CurveData:
    """What a set of curves is fitted to: K dose groups, their samples, each gene's
    means and their unscaled covariance with its scale and degrees of freedom; under
    constant variance also each group's within-group sum of squares."""

    variance: str
    doses: FloatArray  # per dose group
    n_samples: FloatArray  # per dose group
    means: FloatArray  # genes x dose groups
    covariance: FloatArray  # genes x dose groups x dose groups
    covariance_scale: FloatArray  # per gene
    df: FloatArray  # per gene
    within: FloatArray | None  # genes x dose groups under constant variance

    @property
    def n_total(self) -> float:
        return float(self.n_samples.sum())

    def without_group(self, k: int, genes: np.ndarray) -> "_CurveData":
        """The data of some genes without dose group k."""
        keep = np.arange(self.doses.size) != k
        covariance = self.covariance[genes][:, keep][:, :, keep]
        within = None if self.within is None else self.within[genes][:, keep]
        return _CurveData(
            self.variance,
            self.doses[keep],
            self.n_samples[keep],
            self.means[genes][:, keep],
            covariance,
            self.covariance_scale[genes],
            self.df[genes],
            within,
        )


def _constant_variance_data(doses: FloatArray, values: FloatArray) -> _CurveData:
    """The dose groups, means and within-group sums of squares of samples x genes
    values, under constant variance."""
    levels = np.unique(doses)
    group = np.searchsorted(levels, doses)
    n_samples = np.bincount(group, minlength=levels.size).astype(np.float64)
    sums = np.zeros((levels.size, values.shape[1]))
    np.add.at(sums, group, values)
    means = sums / n_samples[:, None]
    # A gene whose values are all equal keeps them as its means, which the division
    # above can move by a unit in the last place, so that it is not fitted as one
    # whose values do not vary.
    constant = np.ptp(values, axis=0) == 0.0
    means[:, constant] = values[0, constant]
    deviations = values - means[group]
    within = np.zeros((levels.size, values.shape[1]))
    np.add.at(within, group, deviations * deviations)
    n_genes = values.shape[1]
    df = float(n_samples.sum() - levels.size)
    total = within.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = total / df if df > 0 else np.full(n_genes, np.nan)
    covariance = np.broadcast_to(
        np.diag(1.0 / n_samples), (n_genes, *[levels.size] * 2)
    )
    return _CurveData(
        CONSTANT,
        levels,
        n_samples,
        np.ascontiguousarray(means.T),
        covariance,
        np.asarray(scale, dtype=np.float64),
        np.full(n_genes, df),
        np.ascontiguousarray(within.T),
    )


# ---------------------------------------------------------------------------------------
# The fits of a set
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _SetFits:
    """The raw fits of a set's models: internal parameters, Q, fitted means,
    convergence and the reasons of fits not made, the model axis in the order of
    ``models``."""

    models: tuple[Model, ...]
    not_fitted: dict[str, str]
    theta: list[FloatArray]  # per model, genes x parameters
    value: FloatArray  # genes x models
    fitted: FloatArray  # genes x models x dose groups
    converged: np.ndarray  # genes x models
    not_fitted_reason: np.ndarray  # genes x models text
    at_bounds: np.ndarray  # genes x models text


def _fit_means(
    data: _CurveData, model_set: str, *, n_threads: int | None = None
) -> DoseCurveFits:
    """Fit a set to the data, with the statistics of every fit, the best model and
    the dose groups that drive turns. The caller has checked that some model of the
    set has a spare dose group."""
    with thread_limit(n_threads):
        raw = _fit_set(data, model_set)
        statistics = _statistics(data, raw)
        best = _best(statistics["aicc"], raw.not_fitted_reason)
        turn = _turn_doses(raw, data.doses)
        driven = _turn_drivers(data, model_set, best, turn)
    parameters = _reported(raw, model_set)
    return DoseCurveFits(
        model_set=model_set,
        models=tuple(m.name for m in raw.models),
        models_not_fitted=MappingProxyType(dict(raw.not_fitted)),
        parameter_names=SET_PARAMETERS[model_set],
        variance=data.variance,
        doses=read_only(data.doses),
        n_samples=read_only(data.n_samples),
        means=read_only(data.means),
        covariance=read_only(data.covariance),
        covariance_scale=read_only(data.covariance_scale),
        df=read_only(data.df),
        within_group_sd=read_only(_within_group_sd(data)),
        parameters=read_only(parameters),
        log_likelihood=read_only(statistics["log_likelihood"]),
        aic=read_only(statistics["aic"]),
        aicc=read_only(statistics["aicc"]),
        akaike_weight=read_only(
            _akaike_weights(statistics["aicc"], raw.not_fitted_reason)
        ),
        lack_of_fit_p_value=read_only(statistics["lack_of_fit_p_value"]),
        max_abs_scaled_residual=read_only(statistics["max_abs_scaled_residual"]),
        parameters_at_bounds=read_only(raw.at_bounds),
        turn_dose=read_only(turn),
        converged=read_only(raw.converged),
        not_fitted_reason=read_only(raw.not_fitted_reason),
        best=read_only(best),
        turn_driven_by_dose=read_only(driven),
    )


def _within_group_sd(data: _CurveData) -> FloatArray:
    """Each dose group's sample SD under constant variance, NaN for a group of one
    sample; NaN in the modes whose means come from a fit."""
    shape = data.means.shape
    if data.within is None:
        return np.full(shape, np.nan)
    replicated = data.n_samples > 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt(data.within / (data.n_samples - 1.0))
    return np.where(replicated, sd, np.nan)


def _models_not_fitted(model_set: str, n_groups: int) -> dict[str, str]:
    """The set's models without a spare dose group at K dose groups, with the
    reason."""
    return {
        m.name: (
            f"no spare dose group: {n_groups} dose groups and {m.n_parameters} "
            "parameters, and a fit needs one group more than its parameters"
        )
        for m in SETS[model_set]
        if n_groups - m.n_parameters < 1
    }


# A standard deviation below this fraction of the means' size is rounding, not
# variation.
_ROUNDING = 1e-12


def _not_fitted_reasons(data: _CurveData, model: Model) -> np.ndarray:
    """Why each gene is not fitted with a model, empty when it is."""
    reasons = np.full(data.means.shape[0], "", dtype=object)
    if data.within is not None:
        flat = (data.within.sum(axis=1) == 0.0) & (np.ptp(data.means, axis=1) == 0.0)
        reasons[flat] = "the values do not vary"
    else:
        # Means whose variance is at their rounding, as a fit of values that do not
        # vary leaves them, vary by nothing.
        rounding = (_ROUNDING * np.maximum(np.abs(data.means).max(axis=1), 1.0)) ** 2
        largest = np.einsum("gkk->gk", data.covariance).max(axis=1)
        variance = data.covariance_scale * largest
        flat = ~(variance > rounding) | ~np.isfinite(data.covariance_scale)
        reasons[flat] = "the variance of the means is 0"
    if model.kind in POSITIVE_MEANS:
        low = (data.means <= 0.0).any(axis=1) & (reasons == "")
        reasons[low] = (
            "a dose-group mean is at or below 0, and the model's means are positive"
        )
    return reasons


def _objective_scales(data: _CurveData) -> FloatArray:
    """The scale of each gene's objective, against which the refinement's decrease is
    small: the within-group sum of squares, or s^2 K."""
    if data.within is not None:
        return np.ascontiguousarray(data.within.sum(axis=1))
    scale = np.where(np.isfinite(data.covariance_scale), data.covariance_scale, 0.0)
    return np.ascontiguousarray(scale * data.doses.size)


def _fit_set(data: _CurveData, model_set: str) -> _SetFits:
    """Fit each model of a set that has a spare dose group to every gene."""
    n_groups = data.doses.size
    not_fitted = _models_not_fitted(model_set, n_groups)
    fitted_models = tuple(m for m in SETS[model_set] if m.name not in not_fitted)
    scale = DoseScale.of(data.doses)
    gap = exponential_sign_gap(scale)
    root = whitening_roots(data.covariance, data.n_samples, data.within is not None)
    shared = root.shape[0] == 1
    scales = _objective_scales(data)
    n_genes = data.means.shape[0]
    n_models = len(fitted_models)
    values = np.full((n_genes, n_models), np.nan)
    fitted = np.full((n_genes, n_models, n_groups), np.nan)
    converged = np.zeros((n_genes, n_models), dtype=bool)
    reasons = np.full((n_genes, n_models), "", dtype=object)
    at_bounds = np.full((n_genes, n_models), "", dtype=object)
    thetas = []
    means = np.ascontiguousarray(data.means)
    for m, model in enumerate(fitted_models):
        kind = model.kind
        reasons[:, m] = _not_fitted_reasons(data, model)
        active = np.ascontiguousarray(reasons[:, m] == "")
        lower, upper = bounds(kind, scale, model_set)
        q = n_linear(kind)
        if kind in (LINE, POLYNOMIAL, QUADRATIC):
            points, inner = np.zeros((0, 1)), 1
            bases = np.zeros((0, n_groups, q))
            usable = np.zeros(0, dtype=bool)
        else:
            points, inner = grid(kind, scale, model_set)
            bases, usable = grid_bases(kind, q, points, data.doses)
            if shared:
                bases = whiten_bases(bases, root[0])
        theta = np.full((n_genes, lower.size), np.nan)
        out_fitted = np.full((n_genes, n_groups), np.nan)
        out_values = np.full(n_genes, np.nan)
        out_converged = np.zeros(n_genes, dtype=bool)
        by_chunks(
            fit_genes_parallel,
            fit_genes_serial,
            (
                kind,
                constraint(kind),
                q,
                points,
                inner,
                bases,
                usable,
                shared,
                data.doses,
                means,
                root,
                lower,
                upper,
                gap,
                scales,
                active,
                theta,
                out_fitted,
                out_values,
                out_converged,
            ),
            n_genes,
        )
        values[:, m] = np.where(active, out_values, np.nan)
        fitted[:, m] = out_fitted
        converged[:, m] = out_converged & active
        theta[~active] = np.nan
        thetas.append(theta)
        at_bounds[:, m] = _at_bounds(model, theta, lower, upper, gap, active)
    return _SetFits(
        fitted_models,
        not_fitted,
        thetas,
        values,
        fitted,
        converged,
        reasons,
        at_bounds,
    )


def _at_bounds(
    model: Model,
    theta: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    gap: float,
    active: np.ndarray,
) -> np.ndarray:
    """The parameters of each fit at a bound, comma-separated; the restricted
    quadratic's b1 or b2 at 0 count, as the restriction holds them there, and so does
    the rate of DRomics' exponential at its smallest size, ``gap``."""
    if model.kind == POLYNOMIAL:
        hit = np.zeros(theta.shape, dtype=bool)
        hit[:, 1:] = theta[:, 1:] == 0.0
    else:
        hit = (theta <= lower) | (theta >= upper)
        if model.kind == DROMICS_EXPONENTIAL:
            hit[:, 2] |= np.abs(theta[:, 2]) <= gap
    names = model.parameters
    out = np.full(theta.shape[0], "", dtype=object)
    for g in np.flatnonzero(active & hit.any(axis=1)):
        out[g] = ", ".join(names[j] for j in np.flatnonzero(hit[g]))
    return out


def _reported(raw: _SetFits, model_set: str) -> FloatArray:
    """The reported parameters of every fit, on the set's parameter axis."""
    names = SET_PARAMETERS[model_set]
    n_genes = raw.value.shape[0]
    out = np.full((n_genes, len(raw.models), len(names)), np.nan)
    for m, model in enumerate(raw.models):
        values = reported(model.kind, raw.theta[m])
        for j, name in enumerate(model.parameters):
            out[:, m, names.index(name)] = values[:, j]
    return out


# ---------------------------------------------------------------------------------------
# Statistics of fits
# ---------------------------------------------------------------------------------------


def _statistics(data: _CurveData, raw: _SetFits) -> dict[str, FloatArray]:
    """Log-likelihoods, AIC, AICc, lack-of-fit p-values and scaled residuals."""
    n_groups = data.doses.size
    n = data.n_total
    sizes = np.array([m.n_parameters for m in raw.models], dtype=np.float64)
    q = raw.value  # genes x models
    residuals = data.means[:, None, :] - raw.fitted  # genes x models x dose groups
    s2 = data.covariance_scale[:, None]
    spare = n_groups - sizes
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if data.within is not None:
            within = data.within.sum(axis=1)[:, None]
            variance = (within + q) / n
            log_likelihood = -0.5 * n * (np.log(2.0 * math.pi * variance) + 1.0)
            errors = np.sqrt(variance)[:, :, None] / np.sqrt(data.n_samples)
        else:
            factor = np.linalg.cholesky(data.covariance)
            logdet = 2.0 * np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(axis=1)
            log_likelihood = (
                -0.5 * n_groups * math.log(2.0 * math.pi)
                - 0.5 * (n_groups * np.log(s2) + logdet[:, None])
                - q / (2.0 * s2)
            )
            diagonal = np.diagonal(data.covariance, axis1=1, axis2=2)
            errors = np.sqrt(s2[:, :, None] * diagonal[:, None, :])
        # The F-test against free dose-group means, in every mode: under constant
        # variance s^2 is the pooled within-group variance, Q / s^2 the pure-error
        # statistic.
        lack = np.asarray(
            f_sf(
                q / (spare * s2),
                np.broadcast_to(spare, q.shape),
                np.broadcast_to(data.df[:, None], q.shape),
            ),
            dtype=np.float64,
        )
        lack = np.where(s2 > 0.0, lack, np.nan)
        scaled = np.abs(residuals / errors)
    k = sizes + 1.0
    aic = -2.0 * log_likelihood + 2.0 * k
    room = n - k - 1.0
    with np.errstate(divide="ignore"):
        correction = np.where(room > 0.0, 2.0 * k * (k + 1.0) / room, np.inf)
    aicc = aic + correction
    fitted = raw.not_fitted_reason == ""
    return {
        "log_likelihood": np.where(fitted, log_likelihood, np.nan),
        "aic": np.where(fitted, aic, np.nan),
        "aicc": np.where(fitted, aicc, np.nan),
        "lack_of_fit_p_value": np.where(fitted, lack, np.nan),
        "max_abs_scaled_residual": np.where(fitted, np.max(scaled, axis=2), np.nan),
    }


def _best(aicc: FloatArray, not_fitted_reason: np.ndarray) -> np.ndarray:
    """Each gene's model of lowest AICc, the first in set order within ``TIE``; -1
    when no AICc is finite or minus infinity."""
    candidates = (not_fitted_reason == "") & (aicc < np.inf)
    low = np.where(candidates, aicc, np.inf).min(axis=1, initial=np.inf)
    with np.errstate(invalid="ignore"):
        tied = candidates & (aicc <= low[:, None] + TIE)
    best = np.argmax(tied, axis=1)
    return np.where(tied.any(axis=1), best, -1).astype(np.int64)


def _akaike_weights(aicc: FloatArray, not_fitted_reason: np.ndarray) -> FloatArray:
    """Each fitted model's share of support among the gene's: exp(-Delta / 2) over
    the sum, Delta the AICc minus the lowest; 0 for an infinite AICc; shared equally
    by the models whose AICc is minus infinity, when any is; NaN where not fitted."""
    fitted = not_fitted_reason == ""
    finite = fitted & (aicc < np.inf)
    low = np.where(finite, aicc, np.inf).min(axis=1, initial=np.inf)
    weights = np.zeros(aicc.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        delta = aicc - low[:, None]
        unbounded = np.isneginf(low)
        delta = np.where(
            unbounded[:, None], np.where(aicc == low[:, None], 0.0, np.inf), delta
        )
        raw = np.where(finite, np.exp(-0.5 * delta), 0.0)
        total = raw.sum(axis=1, keepdims=True)
        weights = np.where(total > 0.0, raw / total, np.nan)
    return np.where(fitted, weights, np.nan)


def _turn_doses(raw: _SetFits, doses: FloatArray) -> FloatArray:
    """The dose of each fit's turn strictly between 0 and the top dose, NaN
    otherwise."""
    top = float(doses[-1])
    out = np.full(raw.value.shape, np.nan)
    for m, model in enumerate(raw.models):
        out[:, m] = turn_doses(model.kind, raw.theta[m], top)
    return out


def _turn_drivers(
    data: _CurveData, model_set: str, best: np.ndarray, turn: FloatArray
) -> FloatArray:
    """For each gene whose best fit turns, the lowest dose whose group, left out,
    leaves a best fit of the whole set that does not turn within the doses kept; NaN
    otherwise, and for every gene when no model that can turn has a spare dose group
    among the K - 1 groups left, where no refit could turn."""
    out = np.full(best.size, np.nan)
    pending = np.flatnonzero(_turning(best, turn))
    if pending.size == 0 or _unchecked_turns(model_set, data.doses.size):
        return out
    # Every refit's set keeps the line, which has a spare group among the K - 1 left.
    for k in range(data.doses.size):
        if pending.size == 0:
            break
        reduced = data.without_group(k, pending)
        refit = _fit_set(reduced, model_set)
        statistics = _statistics(reduced, refit)
        refit_best = _best(statistics["aicc"], refit.not_fitted_reason)
        still = _turning(refit_best, _turn_doses(refit, reduced.doses))
        out[pending[~still]] = data.doses[k]
        pending = pending[still]
    return out
