"""Benchmark doses of fitted dose-response curves, with profile-likelihood bounds.

A fitted curve says what shape a response takes, not at which dose the response
reaches a given size. Regulators read that dose as a benchmark dose (BMD): the dose at
which the curve's change from its level at dose 0 reaches a stated benchmark response
(BMR), with a lower bound (BMDL) and an upper bound (BMDU) (Crump 1984; US EPA 2012).
:func:`benchmark_doses` computes them for every gene of a
:class:`~cytepool.stats.dose_curves.DoseCurveFits`, at one or several
:class:`~cytepool.BenchmarkResponse` values, and :class:`BenchmarkDoses` holds them with
the statistics that the checks of regulatory guidance read: those of BMDS, NTP 2018,
EPA's ETAP 2024 and EFSA 2022. The definitions, the bounds and the checks are in
https://docs.nygen.io/CytePool/statistics/potency#benchmark-doses.

The BMD:
    The lowest dose x above 0 at which |f(x) - f(0)| = R, R the BMR's size, searched
    from a hundredth of the lowest positive dose (DRomics' ``minBMD``) to 50 times the
    top dose. A crossing below the search's lowest dose is given as that dose, as
    DRomics gives it; a crossing above the top dose is given, as BMDS gives it.

The bounds:
    The BMDL and BMDU are the lowest and highest doses b whose profile log-likelihood,
    the largest log-likelihood of a curve whose change at b is R in the BMD's
    direction, lies within half the chi-square(1) quantile at 0.90 of the maximum:
    one-sided 95% bounds each, given the gene's chosen model, and BMDL to BMDU a
    two-sided 90% interval (US EPA 2012 Sec. 2.3.8). Where the BMR is a multiple of
    the SD, the SD moves with the curve in the profile, as BMDS profiles it. Where the
    curves weigh dose-group means with a first-stage variance s^2 on df degrees of
    freedom, an absolute BMR's bounds read the increase of the weighted sum of squares
    over s^2 against F(1, df) at 0.90, which is Fieller's (1954) exact bound for a
    straight line; an SD-based BMR's profile adds the likelihood of s^2 and reads
    chi-square(1).

References:
    Crump KS (1984). A new method for determining allowable daily intakes.
    Fundamental and Applied Toxicology 4, 854-871. doi:10.1016/0272-0590(84)90107-6

    US EPA (2012). Benchmark Dose Technical Guidance. EPA/100/R-12/001, Risk
    Assessment Forum, US Environmental Protection Agency, Washington, DC.

    Fieller EC (1954). Some problems in interval estimation. Journal of the Royal
    Statistical Society, Series B 16, 175-185. doi:10.1111/j.2517-6161.1954.tb00159.x
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._columns import text_column
from cytepool._jit import by_chunks, thread_limit
from cytepool._masked_arrays import read_only
from cytepool._messages import close_matches, listing
from cytepool._options import checked_flag
from cytepool.benchmark_responses import ABSOLUTE, BenchmarkResponse
from cytepool.stats._benchmark_dose_flags import (
    FlagInputs,
    accepted_rows,
    checked_flag_sets,
    failed_checks,
)
from cytepool.stats._benchmark_dose_search import (
    A0,
    BMD,
    CHANGE_GIVEN_VARIANCE,
    CHANGE_IN_SD,
    CHANGE_PROFILED_VARIANCE,
    CROSSED,
    CROSSED_BELOW_RANGE,
    DIRECTION,
    FORM,
    N_EFF,
    NEVER_CROSSED,
    OBJECTIVE_FIT,
    PIECE,
    S2,
    THRESHOLD,
    VALUE,
    bounds_parallel,
    bounds_serial,
    first_crossings,
    scan_doses,
    search_range,
)
from cytepool.stats._dose_curve_models import (
    EXPONENT_MAX,
    EXPONENT_MIN,
    POLYNOMIAL,
    SET_PARAMETERS,
    SETS,
    DoseScale,
    bounds,
    constraint,
    exponential_sign_gap,
    grid,
    internal,
    is_linear,
    mean_values,
    model_of,
    n_linear,
    turn_doses,
)
from cytepool.stats._dose_curve_search import grid_bases, whitening_roots
from cytepool.stats._result_tables import feature_id_column
from cytepool.stats.distributions import chi2_ppf, chi2_sf, f_ppf
from cytepool.stats.dose_curves import TIE, DoseCurveFits
from cytepool.table import Table

__all__ = ["BenchmarkDoses", "benchmark_doses"]

FloatArray = npt.NDArray[np.float64]

LEVEL = 0.95
BOUNDS = "one-sided 95% profile-likelihood bounds"
BEST = "best"

WITHIN = "within the doses"
BELOW = "below the lowest dose"
FLOORED = "below a hundredth of the lowest dose"
ABOVE = "above the top dose"
NOT_REACHED = "not reached"
NO_CURVE = "no curve"
NO_MODEL = "no model chosen"
UNDEFINED = "BMR undefined"
STATUSES = (WITHIN, BELOW, FLOORED, ABOVE, NOT_REACHED, NO_CURVE, NO_MODEL, UNDEFINED)
"""The statuses of a BMD, in the order the repr counts them."""

# A curve's SD at most this share of the largest dose-group mean is rounding alone:
# the values have no spread about the curve.
_ROUNDING = 64.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True, eq=False, repr=False)
class BenchmarkDoses:
    """The benchmark doses of every gene's curve, with their bounds and the
    statistics that the checks of BMDS, NTP 2018, EPA's ETAP 2024 and EFSA 2022 read.

    Rows of the arrays of shape ``(n_bmrs, n_genes)`` are the BMRs, in the order of
    ``benchmark_responses``; columns are genes, in the fits' order. The arrays are
    read-only.
    https://docs.nygen.io/CytePool/statistics/potency#benchmark-doses defines each
    quantity and how to read it.

    Attributes:
        benchmark_responses: The BMRs, as given, then the 1 SD comparison that US EPA
            2012 asks to present beside any other BMR, unless given.
        model_set: The set of the curves: ``"epa"``, ``"bmdexpress"``, ``"ll4"`` or
            ``"dromics"``.
        model: ``"best"``, each gene's model of lowest AICc; a document's choice,
            ``"ntp_2018"``, ``"etap_2024"`` or ``"bmds_2026"``; or the name of the
            model every gene's BMD was computed from.
        gene_models: ``(n_genes,)`` the model of each gene's BMD, empty where none;
            the ``models`` of :class:`~cytepool.stats.dose_curves.DoseCurveFits`
            name the models fitted.
        variance: How the curves weighed the dose-group means: ``"constant"``,
            ``"weighted"`` or ``"moderated"``.
        bounds: Whether bounds were computed; a cell type that supports only
            description gets none.
        lowest_dose: The lowest positive dose of the fits.
        top_dose: The highest dose of the fits.
        bmr_change: The BMR's size R in the values' units, such as log2 units for an
            SD-based BMR.
        bmd: The BMD; NaN where not reached, without a curve or with the BMR
            undefined (``status``).
        bmdl: The lower bound, the lowest dose of the likelihood's region, or under
            NTP's and ETAP's choices of the piece of it that holds the BMD, as BMDS 2
            bounds a BMD; 0 where that reaches the search's lowest dose; NaN without
            bounds, without a BMD, where the values have no spread about the curve,
            or, for every BMR of the gene, where the profile found a likelihood above
            the fit's, at a BMD or at a dose of the search, so that the fit was not
            its maximum.
        bmdu: The upper bound, the highest dose of the same region or piece;
            infinity where that reaches the search's highest dose; NaN as for
            ``bmdl``.
        direction: ``"up"`` or ``"down"``, the direction of the change at the BMD;
            empty without a BMD.
        status: Where the BMD lies: ``"within the doses"``, ``"below the lowest
            dose"``, ``"below a hundredth of the lowest dose"`` (the BMD is given as
            that dose), ``"above the top dose"``, ``"not reached"`` (no crossing up
            to 50 times the top dose), ``"no curve"`` (no model to read: none fitted,
            or none with a finite AICc), ``"no model chosen"`` (a document's choice
            found no model whose BMD it keeps) or ``"BMR undefined"`` (an SD of 0,
            values without spread about the curve, or no vehicle to scale the SD to).
        threshold: The deviance the bounds read: chi-square(1) at 0.90, or F(1, df)
            at 0.90 for an absolute BMR whose curves weigh a first stage's variance;
            NaN without bounds.
        fit_p_value: ``(n_genes,)`` the p-value of BMDS's test 4, the likelihood ratio
            test of each gene's model against a mean per dose group, (df + K) log(1 +
            (K - p) F / df) on a chi-square of K - p degrees of freedom, F the curves'
            F-test of lack of fit: the fit p-value that BMDS, BMDExpress, NTP and ETAP
            read. Every parameter counts in p, as in the curves' AIC, where BMDS's
            default leaves out those at a bound; under a document's choice, p counts
            the mean parameters as that document's program does: those not at a
            bound of BMDS 2's programs for NTP and ETAP, those pybmds does not flag
            as bounded for BMDS. NaN without replicates.
        scaled_residual_near_bmd: The scaled residual of the dose group nearest the
            BMD on the dose scale, the lower dose on a tie (BMDS's residual of
            interest).
        scaled_residual_at_vehicle: ``(n_genes,)`` the scaled residual of the vehicle
            group.
        vehicle_sd_ratio: ``(n_genes,)`` under constant variance, the curve's SD over
            the vehicle replicates' sample SD (at least 1e-6), BMDS's check of the
            modelled SD; NaN otherwise.
        constant_variance_p_value: ``(n_genes,)`` under constant variance, the
            likelihood ratio test of one variance against a variance per dose group
            (BMDS's test 2), on K - 1 degrees of freedom; NaN otherwise, and where a
            group has fewer than two samples or no spread.
        model_choice: ``(n_genes,)`` under a document's choice, why each gene's
            model was chosen, such as ``"lowest AIC"``; empty otherwise.
        bmd_vehicle_sd: ``(n_genes,)`` under ETAP's choice, the BMD of the first
            BMR's multiple of the vehicle replicates' sample SD, ETAP's wording of
            its BMR, without bounds; NaN otherwise.
        feature_ids: ``(n_genes,)`` the genes' ids that :meth:`table` writes, from
            :meth:`DoseResponseResult.benchmark_doses
            <cytepool.DoseResponseResult.benchmark_doses>`; None from
            :func:`benchmark_doses`, whose table numbers the genes.
    """

    benchmark_responses: tuple[BenchmarkResponse, ...]
    model_set: str
    model: str
    gene_models: np.ndarray
    variance: str
    bounds: bool
    lowest_dose: float
    top_dose: float
    bmr_change: FloatArray
    bmd: FloatArray
    bmdl: FloatArray
    bmdu: FloatArray
    direction: np.ndarray
    status: np.ndarray
    threshold: FloatArray
    fit_p_value: FloatArray
    scaled_residual_near_bmd: FloatArray
    scaled_residual_at_vehicle: FloatArray
    vehicle_sd_ratio: FloatArray
    constant_variance_p_value: FloatArray
    model_choice: np.ndarray
    bmd_vehicle_sd: FloatArray
    feature_ids: np.ndarray | None = None

    def accepted(self, flag_set: str) -> npt.NDArray[np.bool_]:
        """Whether each BMD passes a document's acceptance: a BMD whose fit p-value
        is a number and that fails none of the document's checks but those it keeps
        aside, such as NTP's flag of a BMD more than 10-fold below the lowest dose
        (https://docs.nygen.io/CytePool/statistics/potency#rule-sets). The checks
        skip what a row lacks, and acceptance asks for all a criterion reads.

        Args:
            flag_set: ``"ntp_2018"`` or ``"etap_2024"``; BMDS's bins and EFSA's
                indicators accept no gene and are refused.

        Returns:
            ``(n_bmrs, n_genes)``, rows the BMRs in order.

        Raises:
            ValueError: If ``flag_set`` is not ``"ntp_2018"`` or ``"etap_2024"``,
                or the BMDs have no bounds, which both documents' criteria read.
            TypeError: If ``flag_set`` is not text.
        """
        if not isinstance(flag_set, str):
            raise TypeError(f"flag_set must be text, got {type(flag_set).__name__}")
        if flag_set not in (NTP_2018, ETAP_2024):
            raise ValueError(
                "accepted reads a document's acceptance of a gene's BMD, which "
                f"'ntp_2018' and 'etap_2024' state; got {flag_set!r}"
                f"{close_matches(flag_set, (NTP_2018, ETAP_2024))}; BMDS's bins "
                "('bmds_2026') and EFSA's indicators ('efsa_2022') accept no gene"
            )
        kept = accepted_rows(flag_set, self._flag_inputs())
        return read_only(kept.reshape(self.bmd.shape))

    def table(self, *, feature_ids: Any = None, flags: Any = None) -> Table:
        """Return the benchmark doses as a table, one row per BMR and gene.

        The BMRs in order, each with every gene in order: ``feature_id``, ``bmr`` (the
        BMR's label), ``model``, under a document's choice ``model_choice`` and
        ``fit_p_value``, then ``direction``, ``bmr_change``, ``bmd``, ``bmdl``,
        ``bmdu``, ``bmd_status``, then ``flags_<set>`` for each set of ``flags``: the
        checks the row fails, joined by "; ", empty when none
        (https://docs.nygen.io/CytePool/statistics/potency#benchmark-dose-flags).

        Args:
            feature_ids: One id per gene, as text; by default the result's
                ``feature_ids``, or ``feature_0``, ``feature_1``, ... without them.
            flags: The checks of BMDS, NTP, ETAP or EFSA, as a set's name or a list
                of names: ``"bmds_2026"``, ``"ntp_2018"``, ``"etap_2024"`` or
                ``"efsa_2022"``; None for none.

        Raises:
            ValueError: If ``feature_ids`` does not give one id per gene, or a set is
                unknown or repeated.
            TypeError: If an id or a set is not text, or ``feature_ids`` is a numpy
                masked array.
        """
        sets = checked_flag_sets(flags)
        n_bmrs, n_genes = self.bmd.shape
        given = self.feature_ids if feature_ids is None else feature_ids
        ids = feature_id_column(given, n_genes)
        genes = np.tile(np.arange(n_genes), n_bmrs)
        labels = np.repeat([bmr.label for bmr in self.benchmark_responses], n_genes)
        columns: dict[str, Any] = {
            "feature_id": ids[genes],
            "bmr": text_column(labels.tolist()),
            "model": text_column(self.gene_models[genes].tolist()),
        }
        if self.model in DOCUMENTS:
            columns["model_choice"] = text_column(self.model_choice[genes].tolist())
            columns["fit_p_value"] = self.fit_p_value[genes]
        columns |= {
            "direction": text_column(self.direction.ravel().tolist()),
            "bmr_change": self.bmr_change.ravel(),
            "bmd": self.bmd.ravel(),
            "bmdl": self.bmdl.ravel(),
            "bmdu": self.bmdu.ravel(),
            "bmd_status": text_column(self.status.ravel().tolist()),
        }
        for flag_set in sets:
            columns[f"flags_{flag_set}"] = text_column(
                failed_checks(flag_set, self._flag_inputs())
            )
        return Table(columns)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return (_frozen_benchmark_doses, (values,))

    def _flag_inputs(self) -> FlagInputs:
        n_bmrs = self.bmd.shape[0]

        def rows(values: FloatArray) -> FloatArray:
            return np.tile(values, n_bmrs)

        return FlagInputs(
            bmd=self.bmd.ravel(),
            bmdl=self.bmdl.ravel(),
            bmdu=self.bmdu.ravel(),
            bounds=self.bounds,
            lowest_dose=self.lowest_dose,
            top_dose=self.top_dose,
            fit_p_value=rows(self.fit_p_value),
            residual_near_bmd=self.scaled_residual_near_bmd.ravel(),
            residual_at_vehicle=rows(self.scaled_residual_at_vehicle),
            vehicle_sd_ratio=rows(self.vehicle_sd_ratio),
            constant_variance_p_value=rows(self.constant_variance_p_value),
        )

    def __repr__(self) -> str:
        n_genes = self.bmd.shape[1]
        bounds = BOUNDS if self.bounds else "no bounds"
        lines = [
            f"BenchmarkDoses ({self.model_set}, {self.model} model, {self.variance} "
            "variance): "
            f"{n_genes:,} gene{'' if n_genes == 1 else 's'}; {bounds}"
        ]
        for i, bmr in enumerate(self.benchmark_responses):
            lines.append(f"  {bmr.label}: {_status_counts(self.status[i])}")
        missing = int(np.count_nonzero(self._bounds_missing()))
        if missing:
            lines.append(
                f"  bounds not computed for {missing:,} BMD"
                f"{'' if missing == 1 else 's'}: the profile found a likelihood above "
                "the fit's, or the values have no spread about the curve"
            )
        return "\n".join(lines)

    def _bounds_missing(self) -> np.ndarray:
        """Rows with a BMD whose bounds were asked for and could not be computed."""
        return np.asarray(self.bounds & np.isfinite(self.bmd) & np.isnan(self.bmdl))


def _frozen_benchmark_doses(values: dict[str, Any]) -> BenchmarkDoses:
    """BenchmarkDoses of read-only arrays, as benchmark_doses builds them; pickled
    results are rebuilt by it."""
    frozen: dict[str, Any] = {
        name: read_only(value) if isinstance(value, np.ndarray) else value
        for name, value in values.items()
    }
    return BenchmarkDoses(**frozen)


def _status_counts(status: np.ndarray) -> str:
    """The count of each status, in the order of ``STATUSES``, for printouts."""
    parts = [
        f"{int(np.count_nonzero(status == name)):,} {name}"
        for name in STATUSES
        if np.any(status == name)
    ]
    return ", ".join(parts) or "none"


def _benchmark_record(found: BenchmarkDoses | None) -> dict[str, Any] | None:
    """The record of a cell type's benchmark doses: the BMRs, the bounds or why there
    are none, the distribution each BMR's bounds read, the doses searched and the BMDs
    whose bounds could not be computed; None without them."""
    if found is None:
        return None
    low, high = search_range(found.lowest_dose, found.top_dose)
    return {
        "bmrs": [bmr.to_dict() for bmr in found.benchmark_responses],
        "bounds": (
            BOUNDS if found.bounds else "none: the cell type supports only description"
        ),
        "thresholds": {
            bmr.label: _threshold_name(found.variance, bmr)
            for bmr in found.benchmark_responses
        },
        "doses_searched": [low, high],
        "n_bounds_not_computed": int(np.count_nonzero(found._bounds_missing())),
    }


def _threshold_name(variance: str, bmr: BenchmarkResponse) -> str:
    """The distribution whose 0.90 quantile the bounds of a BMR read."""
    if variance == "constant" or bmr.kind != ABSOLUTE:
        return "chi-square(1) at 0.90"
    return "F(1, df) at 0.90, df the degrees of freedom of each gene's variance"


def benchmark_doses(
    fits: DoseCurveFits,
    bmr: BenchmarkResponse | Sequence[BenchmarkResponse],
    *,
    model: str = BEST,
    bounds: bool = True,
    n_threads: int | None = None,
) -> BenchmarkDoses:
    """Compute the benchmark dose of every gene's curve, with its bounds.

    The BMD is the lowest dose at which the curve's change from its level at dose 0
    reaches the BMR, and the BMDL and BMDU are one-sided 95% bounds by profile
    likelihood, given each gene's chosen model, so that BMDL to BMDU is a two-sided
    90% interval (Crump 1984; US EPA 2012); the bounds do not carry the uncertainty
    of the model's choice
    (https://docs.nygen.io/CytePool/statistics/potency#potency-limits). The 1 SD BMD,
    which US EPA 2012 asks to present beside any other BMR, is added unless given.
    Results do not depend on the number of threads.
    https://docs.nygen.io/CytePool/statistics/potency#benchmark-doses defines the
    BMD, its bounds and where each lies.

    Args:
        fits: Curves of named models, from
            :func:`~cytepool.stats.dose_curves.fit_dose_curves` or
            :meth:`DoseResponseResult.curve_fits <cytepool.DoseResponseResult.curve_fits>`.
        bmr: A :class:`~cytepool.BenchmarkResponse`, or a list of them with distinct
            labels.
        model: ``"best"`` (default), each gene's model of lowest AICc; the name of
            a model of the fits' set, for every gene it was fitted to; or a
            document's choice: ``"ntp_2018"`` or ``"etap_2024"`` on the set
            ``"bmdexpress"``, the lowest AIC as BMDS 2 counts it among the models
            whose BMD has both bounds, ETAP's leaving out a Hill fit whose k is below
            a third of the lowest dose; ``"bmds_2026"`` on the set ``"epa"``, BMDS's
            recommended model as pybmds 25.2 chooses it among linear, polynomial 2,
            power, Hill, exponential 3 and 5, which ``dose_response`` makes with
            ``selection="bmds_2026"``, and with ``rules`` the documents' choices.
            Under NTP's and ETAP's choices the bounds end at the piece of the
            likelihood's region that holds the BMD, as BMDS 2's do. A document's
            choice reads the first BMR, keeps that model for the other BMRs, needs
            constant variance and bounds, and computes the bounds of the models it
            passes over, so it costs more.
        bounds: False gives BMDs without bounds, as for curves of dose groups
            without replicates.
        n_threads: The most threads to use; None keeps numba's setting.

    Returns:
        The BMDs, with their bounds and the statistics the checks of regulatory
        guidance read. Its ``table(flags=...)`` lists the checks of BMDS, NTP, ETAP
        or EFSA that each BMD fails.

    Raises:
        TypeError: If ``fits`` is not a ``DoseCurveFits``, ``bmr`` not a
            ``BenchmarkResponse`` or a list of them, ``model`` not a str or ``bounds``
            not True or False.
        ValueError: If ``bmr`` is empty or repeats a label, ``model`` names no model
            of the set or one not fitted, or a document's choice for fits of another
            set, of variance other than constant or without bounds, or bounds are
            asked for curves without replicates.
    """
    if not isinstance(fits, DoseCurveFits):
        raise TypeError(f"fits must be a DoseCurveFits, got {type(fits).__name__}")
    responses = _with_comparison(_checked_responses(bmr))
    with_bounds = checked_flag(bounds, "bounds")
    document = _checked_document(fits, model, with_bounds)
    if document is None:
        chosen = _chosen_models(fits, model)
        why = np.full(fits.means.shape[0], "", dtype=object)
    elif document == BMDS_2026:
        with thread_limit(n_threads):
            chosen, why = _bmds_choice(fits, responses[0])
    else:
        with thread_limit(n_threads):
            chosen, why = _bmdexpress_choice(fits, responses[0], document)
    if with_bounds and not np.all(fits.df > 0):
        raise ValueError(
            "bounds need replicates within dose groups: these fits have no degrees "
            "of freedom for the variance (df 0), as for a cell type that supports "
            "only description; pass bounds=False for BMDs without bounds"
        )
    genes = _GeneFits.of(fits, chosen)
    with thread_limit(n_threads):
        # BMDS 2, which NTP and ETAP run, bounds a BMD by the piece of the region
        # that holds it.
        piece = document in (NTP_2018, ETAP_2024)
        rows = [
            _for_response(fits, genes, bmr_, with_bounds, piece) for bmr_ in responses
        ]
    stacked = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    # A fit that is not the profile's maximum at one BMR is not the maximum of the
    # likelihood: no bounds are measured from it at any BMR.
    missing = with_bounds & np.isfinite(stacked["bmd"]) & np.isnan(stacked["bmdl"])
    unbounded = np.any(missing, axis=0)
    stacked["bmdl"][:, unbounded] = np.nan
    stacked["bmdu"][:, unbounded] = np.nan
    near = np.stack([_residuals_near(fits, genes, row["bmd"]) for row in rows])
    names = np.array([fits.models[m] if m >= 0 else "" for m in chosen], dtype=object)
    fit_p = genes.fit_p_value
    if document is not None:
        # Test 4 on K minus the parameters that the document's program counts.
        counted = (
            _pybmds_counted(fits) if document == BMDS_2026 else _bmds2_counted(fits)
        )
        every = _test_4_p_values(fits, counted)
        picked = chosen >= 0
        fit_p = np.full(chosen.size, np.nan)
        fit_p[picked] = every[np.flatnonzero(picked), chosen[picked]]
        fitted = (fits.not_fitted_reason != "").sum(axis=1) < len(fits.models)
        stacked["status"][:, ~picked & fitted] = NO_MODEL
    return BenchmarkDoses(
        benchmark_responses=tuple(responses),
        model_set=fits.model_set,
        model=model,
        gene_models=read_only(names),
        variance=fits.variance,
        bounds=with_bounds,
        lowest_dose=float(fits.doses[fits.doses > 0.0][0]),
        top_dose=float(fits.doses[-1]),
        bmr_change=read_only(stacked["size"]),
        bmd=read_only(stacked["bmd"]),
        bmdl=read_only(stacked["bmdl"]),
        bmdu=read_only(stacked["bmdu"]),
        direction=read_only(stacked["direction"]),
        status=read_only(stacked["status"]),
        threshold=read_only(stacked["threshold"]),
        fit_p_value=read_only(fit_p),
        scaled_residual_near_bmd=read_only(near),
        scaled_residual_at_vehicle=read_only(
            genes.scaled_residuals[:, 0]
            if fits.doses[0] == 0.0
            else np.full(fits.means.shape[0], np.nan)
        ),
        vehicle_sd_ratio=read_only(_vehicle_sd_ratio(fits, genes)),
        constant_variance_p_value=read_only(_constant_variance_test(fits)),
        model_choice=read_only(why),
        bmd_vehicle_sd=read_only(
            _vehicle_sd_bmds(fits, genes, responses[0])
            if document == ETAP_2024
            else np.full(chosen.size, np.nan)
        ),
    )


# ---------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------


def _checked_responses(bmr: Any) -> list[BenchmarkResponse]:
    responses = [bmr] if isinstance(bmr, BenchmarkResponse) else bmr
    if isinstance(responses, str | bytes) or not isinstance(responses, Sequence):
        raise TypeError(
            "bmr must be a BenchmarkResponse or a list of them, got "
            f"{type(bmr).__name__}; build one with BenchmarkResponse.sd(1.349) or "
            "BenchmarkResponse.absolute(change)"
        )
    responses = list(responses)
    if not responses:
        raise ValueError("bmr must hold at least one BenchmarkResponse")
    for item in responses:
        if not isinstance(item, BenchmarkResponse):
            raise TypeError(
                f"bmr must hold BenchmarkResponse values, got {type(item).__name__}"
            )
    labels = [item.label for item in responses]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise ValueError(
            f"bmr repeats {listing([repr(label) for label in repeated])}; give each "
            "BMR once, with values that differ in their first 6 significant digits"
        )
    return responses


def _with_comparison(responses: list[BenchmarkResponse]) -> list[BenchmarkResponse]:
    """The BMRs with the 1 SD comparison (US EPA 2012 Sec. 2.2.2) added last, unless
    a BMR of that label is given."""
    one_sd = BenchmarkResponse.sd(1.0)
    labels = [bmr.label for bmr in responses]
    return responses if one_sd.label in labels else [*responses, one_sd]


def _chosen_models(fits: DoseCurveFits, model: Any) -> np.ndarray:
    """The position in ``fits.models`` of each gene's model, -1 where none."""
    if not isinstance(model, str):
        raise TypeError(f"model must be a str, got {type(model).__name__}")
    if model == BEST:
        return np.asarray(fits.best, dtype=np.int64)
    names = [m.name for m in SETS[fits.model_set]]
    if model not in names:
        raise ValueError(
            f"model must be 'best' or a model of the set {fits.model_set!r}, "
            f"{listing(names)}; got {model!r}{close_matches(model, names)}"
        )
    if model not in fits.models:
        raise ValueError(
            f"model {model!r} was not fitted: "
            f"{fits.models_not_fitted.get(model, 'it is not among the fits')}"
        )
    position = fits.models.index(model)
    fitted = fits.not_fitted_reason[:, position] == ""
    return np.where(fitted, position, -1).astype(np.int64)


# ---------------------------------------------------------------------------------------
# Each gene's curve and likelihood
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _GeneFits:
    """Each gene's chosen model: its kind, internal parameters, the whitening factor
    of the means' covariance, the weighted sum of squares and, under constant
    variance, the curve's maximum-likelihood variance."""

    chosen: np.ndarray  # per gene, the position in fits.models, -1 for none
    kinds: np.ndarray  # per gene, the kind, -1 for none
    theta: list[FloatArray]  # per gene, internal parameters (empty without a model)
    roots: FloatArray  # one factor under constant variance, else one per gene
    q_min: FloatArray  # per gene
    variance: FloatArray  # per gene, (S_w + Q) / n; NaN in the plug-in modes
    fit_p_value: FloatArray  # per gene, BMDS's test 4
    scaled_residuals: FloatArray  # genes x dose groups
    scale: FloatArray  # per gene, the largest |dose-group mean|

    @classmethod
    def of(cls, fits: DoseCurveFits, chosen: np.ndarray) -> "_GeneFits":
        n_genes, n_groups = fits.means.shape
        names = SET_PARAMETERS[fits.model_set]
        kinds = np.full(n_genes, -1, dtype=np.int64)
        theta: list[FloatArray] = [np.zeros(0)] * n_genes
        fitted = np.full((n_genes, n_groups), np.nan)
        spare = np.full(n_genes, np.nan)
        for m, name in enumerate(fits.models):
            genes = np.flatnonzero(chosen == m)
            if genes.size == 0:
                continue
            model = model_of(fits.model_set, name)
            columns = [names.index(p) for p in model.parameters]
            values = np.ascontiguousarray(
                internal(model.kind, fits.parameters[genes][:, m, columns])
            )
            kinds[genes] = model.kind
            fitted[genes] = mean_values(model.kind, values, fits.doses)
            spare[genes] = n_groups - model.n_parameters
            for i, g in enumerate(genes):
                theta[g] = values[i]
        roots = whitening_roots(
            fits.covariance, fits.n_samples, fits.variance == "constant"
        )
        # Test 4 exists where the curves' F-test of lack of fit does: not for cell
        # types that support only description.
        rows = np.arange(n_genes)
        tested = (chosen >= 0) & np.isfinite(
            fits.lack_of_fit_p_value[rows, np.maximum(chosen, 0)]
        )
        residuals = fits.means - fitted
        root_for = (
            roots
            if roots.shape[0] == n_genes
            else np.broadcast_to(roots, (n_genes, n_groups, n_groups))
        )
        whitened = np.einsum("gij,gj->gi", root_for, residuals)
        q_min = np.sum(whitened * whitened, axis=1)
        # Scaled residuals as the fits define them: sigma / sqrt(n_k) under constant
        # variance, sigma the curve's maximum-likelihood SD; s sqrt(C_kk) otherwise.
        if fits.variance == "constant":
            n = float(fits.n_samples.sum())
            variance = (_within_sum(fits) + q_min) / n
            errors = np.sqrt(variance)[:, None] / np.sqrt(fits.n_samples)[None, :]
        else:
            variance = np.full(n_genes, np.nan)
            diagonal = np.diagonal(fits.covariance, axis1=1, axis2=2)
            errors = np.sqrt(fits.covariance_scale[:, None] * diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = residuals / errors
        return cls(
            chosen,
            kinds,
            theta,
            roots,
            q_min,
            variance,
            np.where(tested, _test_4(fits, q_min, spare), np.nan),
            scaled,
            np.max(np.abs(fits.means), axis=1),
        )


def _test_4(fits: DoseCurveFits, q_min: FloatArray, spare: FloatArray) -> FloatArray:
    """BMDS's test 4, the likelihood ratio of a gene's curve against a mean per dose
    group: (df + K) log(1 + Q / (df s^2)), with s^2 on df degrees of freedom, on a
    chi-square of K - p degrees of freedom; Q / s^2 at infinite df. Under constant
    variance df s^2 is the within-group sum of squares and df + K the samples, BMDS's
    form; in the plug-in modes it is the likelihood ratio of the likelihood that
    holds s^2 with the curve. NaN without a curve or replicates."""
    n_groups = fits.doses.size
    s2, df = fits.covariance_scale, fits.df
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        statistic = np.where(
            np.isfinite(df),
            (df + n_groups) * np.log1p(q_min / (df * s2)),
            q_min / s2,
        )
    ok = np.isfinite(statistic) & (spare > 0) & (df > 0) & (s2 > 0)
    p = np.asarray(
        chi2_sf(np.where(ok, statistic, 0.0), np.where(ok, spare, 1.0)),
        dtype=np.float64,
    )
    return np.where(ok, p, np.nan)


def _within_sum(fits: DoseCurveFits) -> FloatArray:
    """The within-group sum of squares of each gene under constant variance: the
    pooled variance times its degrees of freedom, 0 without replicates."""
    with np.errstate(invalid="ignore"):
        total = fits.covariance_scale * fits.df
    return np.asarray(np.where(fits.df > 0, total, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class _Likelihood:
    """What the profile of one BMR reads for each gene: the form, the constants of
    the likelihood, the BMR's value in the search (R, or the SD multiple scaled to
    a vehicle replicate), its size R (0 where the values have no spread about the
    curve and the BMR is SD-based), the objective at the fit, the threshold, and
    whether the likelihood's variance exceeds rounding, so that bounds exist."""

    forms: np.ndarray
    a0: FloatArray
    n_eff: FloatArray
    s2: FloatArray
    values: FloatArray
    sizes: FloatArray
    objective_fit: FloatArray
    thresholds: FloatArray
    spread: np.ndarray


def _likelihood(
    fits: DoseCurveFits, genes: _GeneFits, bmr: BenchmarkResponse
) -> _Likelihood:
    n_genes, n_groups = fits.means.shape
    chi_square = float(chi2_ppf(2.0 * LEVEL - 1.0, 1.0))
    q_min = genes.q_min
    ones = np.ones(n_genes)
    with np.errstate(divide="ignore", invalid="ignore"):
        if fits.variance == "constant":
            within = _within_sum(fits)
            n = float(fits.n_samples.sum())
            sigma2 = genes.variance
            spread = _spread(sigma2, genes)
            if bmr.kind == ABSOLUTE:
                return _Likelihood(
                    np.full(n_genes, CHANGE_PROFILED_VARIANCE),
                    within,
                    n * ones,
                    ones,
                    bmr.value * ones,
                    bmr.value * ones,
                    n * np.log(within + q_min),
                    chi_square * ones,
                    spread,
                )
            return _Likelihood(
                np.full(n_genes, CHANGE_IN_SD),
                within,
                n * ones,
                ones,
                bmr.value * ones,
                np.where(spread, bmr.value * np.sqrt(sigma2), 0.0),
                n * np.log(sigma2) + n,
                chi_square * ones,
                spread,
            )
        s2 = fits.covariance_scale
        df = fits.df
        if bmr.kind == ABSOLUTE:
            return _Likelihood(
                np.full(n_genes, CHANGE_GIVEN_VARIANCE),
                np.zeros(n_genes),
                np.zeros(n_genes),
                s2,
                bmr.value * ones,
                bmr.value * ones,
                q_min / s2,
                np.asarray(f_ppf(2.0 * LEVEL - 1.0, 1.0, df), dtype=np.float64),
                _spread(s2, genes),
            )
        # The SD of a vehicle replicate: sigma sqrt(n_0 C_00), C_00 the vehicle
        # mean's unscaled variance; no vehicle, no SD to scale to.
        if fits.doses[0] == 0.0:
            vehicle = np.sqrt(fits.n_samples[0] * fits.covariance[:, 0, 0])
        else:
            vehicle = np.full(n_genes, np.nan)
        multiple = bmr.value * vehicle
        finite = np.isfinite(df)
        a0 = np.where(finite, df * s2, 0.0)
        n_eff = np.where(finite, df + n_groups, 0.0)
        sigma2 = np.where(finite, (a0 + q_min) / n_eff, s2)
        spread = _spread(sigma2, genes)
        sizes = np.where(spread, multiple * np.sqrt(sigma2), 0.0)
        return _Likelihood(
            np.where(finite, CHANGE_IN_SD, CHANGE_GIVEN_VARIANCE).astype(np.int64),
            a0,
            n_eff,
            s2,
            np.where(finite, multiple, sizes),
            sizes,
            np.where(finite, n_eff * np.log(sigma2) + n_eff, q_min / s2),
            chi_square * ones,
            spread,
        )


def _spread(variance: FloatArray, genes: _GeneFits) -> np.ndarray:
    """Whether each gene's SD in the likelihood exceeds rounding: more than 64
    machine epsilons of its largest dose-group mean."""
    with np.errstate(invalid="ignore"):
        return np.asarray(np.sqrt(variance) > _ROUNDING * genes.scale)


# ---------------------------------------------------------------------------------------
# The BMDs and bounds of one BMR
# ---------------------------------------------------------------------------------------


def _for_response(
    fits: DoseCurveFits,
    genes: _GeneFits,
    bmr: BenchmarkResponse,
    with_bounds: bool,
    piece: bool = False,
) -> dict[str, np.ndarray]:
    """Each gene's BMD at one BMR, with its bounds when asked: the ends of the
    profile's region, or with ``piece`` of the piece of it that holds the BMD."""
    n_genes = fits.means.shape[0]
    likelihood = _likelihood(fits, genes, bmr)
    scale = DoseScale.of(fits.doses)
    low, high = search_range(scale.lowest, scale.top)
    scan = scan_doses(scale.lowest, scale.top)
    bmd = np.full(n_genes, np.nan)
    bmdl = np.full(n_genes, np.nan)
    bmdu = np.full(n_genes, np.nan)
    direction = np.zeros(n_genes)
    how = np.full(n_genes, -1, dtype=np.int64)
    sizes = likelihood.sizes
    defined = np.isfinite(sizes) & (sizes > 0.0)
    for kind in np.unique(genes.kinds[genes.kinds >= 0]):
        rows = np.flatnonzero((genes.kinds == kind) & defined)
        if rows.size == 0:
            continue
        kind = int(kind)
        theta = np.ascontiguousarray(np.stack([genes.theta[g] for g in rows]))
        turns = np.ascontiguousarray(turn_doses(kind, theta, high))
        found = np.full(rows.size, np.nan)
        signs = np.zeros(rows.size)
        codes = np.zeros(rows.size, dtype=np.int64)
        first_crossings(
            kind,
            theta,
            np.ascontiguousarray(sizes[rows]),
            turns,
            low,
            high,
            found,
            signs,
            codes,
        )
        bmd[rows], direction[rows], how[rows] = found, signs, codes
        if with_bounds:
            # Values without spread about the curve leave the likelihood without a
            # variance to bound with.
            reached = (codes != NEVER_CROSSED) & likelihood.spread[rows]
            lower_, upper_ = _bounds_of_kind(
                fits,
                genes,
                likelihood,
                kind,
                rows[reached],
                theta[reached],
                found[reached],
                signs[reached],
                scan,
                piece,
            )
            bmdl[rows[reached]], bmdu[rows[reached]] = lower_, upper_
    status = _statuses(genes.chosen, defined, how, bmd, scale)
    thresholds = np.where(with_bounds & np.isfinite(bmd), likelihood.thresholds, np.nan)
    return {
        "size": np.where(defined, sizes, np.nan),
        "bmd": bmd,
        "bmdl": bmdl,
        "bmdu": bmdu,
        "direction": np.array(
            ["up" if d > 0 else "down" if d < 0 else "" for d in direction],
            dtype=object,
        ),
        "status": status,
        "threshold": thresholds,
    }


def _bounds_of_kind(
    fits: DoseCurveFits,
    genes: _GeneFits,
    likelihood: _Likelihood,
    kind: int,
    rows: np.ndarray,
    theta: FloatArray,
    bmd: FloatArray,
    signs: FloatArray,
    scan: FloatArray,
    piece: bool,
) -> tuple[FloatArray, FloatArray]:
    """The BMDL and BMDU of the genes of one kind of model."""
    if rows.size == 0:
        return np.zeros(0), np.zeros(0)
    model, points, inner, phi_fit = _search_model(fits, kind, theta)
    bases, usable = grid_bases(kind, model[2], points, fits.doses)
    roots = genes.roots if genes.roots.shape[0] == 1 else genes.roots[rows]
    found = np.full((rows.size, 2), np.nan)
    by_chunks(
        bounds_parallel,
        bounds_serial,
        (
            model,
            points,
            inner,
            usable,
            np.ascontiguousarray(bases),
            scan,
            _grid_changes(kind, model[2], points, scan),
            fits.doses,
            np.ascontiguousarray(fits.means[rows]),
            np.ascontiguousarray(roots),
            _gene_numbers(likelihood, rows, signs, bmd, piece),
            phi_fit,
            found,
        ),
        rows.size,
    )
    return found[:, 0], found[:, 1]


def _search_model(
    fits: DoseCurveFits, kind: int, theta: FloatArray
) -> tuple[tuple[Any, ...], FloatArray, int, FloatArray]:
    """What the search reads of a kind of model: its fixed description (kind, the
    constraint of its linear parameters, their number, the positions of its
    nonlinear parameters, their bounds and the gap of DRomics' exponential rate),
    the grid of its nonlinear parameters with the grid's inner size, and each
    gene's fitted nonlinear parameters, from the genes' internal parameters
    ``theta``. The straight line and the quadratic have no nonlinear parameters:
    their grid is one point."""
    scale = DoseScale.of(fits.doses)
    q = n_linear(kind)
    p = theta.shape[1]
    nonlinear = [j for j in range(p) if not is_linear(kind, j)]
    index = np.array(nonlinear if p > q else [], dtype=np.int64)
    if index.size:
        points, inner = grid(kind, scale, fits.model_set)
        lower, upper = bounds(kind, scale, fits.model_set)
        lower, upper = lower[index], upper[index]
        phi_fit = theta[:, index]
    else:
        points, inner = np.zeros((1, 1)), 1
        lower = upper = np.zeros(0)
        phi_fit = np.zeros((theta.shape[0], 1))
    model = (
        kind,
        int(constraint(kind)),
        q,
        index,
        np.ascontiguousarray(lower, dtype=np.float64),
        np.ascontiguousarray(upper, dtype=np.float64),
        float(exponential_sign_gap(scale)),
    )
    return (
        model,
        np.ascontiguousarray(points, dtype=np.float64),
        inner,
        np.ascontiguousarray(phi_fit, dtype=np.float64),
    )


def _grid_changes(
    kind: int, q: int, points: FloatArray, doses: FloatArray
) -> FloatArray:
    """The linear columns of every grid point at each dose minus at dose 0, ``(n_points,
    n_doses, q)``; NaN where not finite."""
    at_doses, _ = grid_bases(kind, q, points, doses)
    at_zero, _ = grid_bases(kind, q, points, np.zeros(1))
    change = at_doses - at_zero
    change[~np.isfinite(change)] = np.nan
    return np.ascontiguousarray(change)


def _gene_numbers(
    likelihood: _Likelihood,
    rows: np.ndarray,
    signs: FloatArray,
    bmd: FloatArray,
    piece: bool = False,
) -> FloatArray:
    """Each gene's numbers, as the search reads them: the columns ``FORM`` to
    ``PIECE``, with ``piece`` for bounds that end at the piece of the likelihood's
    region that holds the BMD."""
    numbers = np.empty((rows.size, PIECE + 1))
    numbers[:, FORM] = likelihood.forms[rows]
    numbers[:, VALUE] = likelihood.values[rows]
    numbers[:, DIRECTION] = signs
    numbers[:, A0] = likelihood.a0[rows]
    numbers[:, N_EFF] = likelihood.n_eff[rows]
    numbers[:, S2] = likelihood.s2[rows]
    numbers[:, BMD] = bmd
    numbers[:, OBJECTIVE_FIT] = likelihood.objective_fit[rows]
    numbers[:, THRESHOLD] = likelihood.thresholds[rows]
    numbers[:, PIECE] = 1.0 if piece else 0.0
    return numbers


def _statuses(
    chosen: np.ndarray,
    defined: np.ndarray,
    how: np.ndarray,
    bmd: FloatArray,
    scale: DoseScale,
) -> np.ndarray:
    status = np.full(how.size, NO_CURVE, dtype=object)
    curve = chosen >= 0
    status[curve & ~defined] = UNDEFINED
    status[curve & defined & (how == NEVER_CROSSED)] = NOT_REACHED
    status[curve & defined & (how == CROSSED_BELOW_RANGE)] = FLOORED
    found = curve & defined & (how == CROSSED)
    with np.errstate(invalid="ignore"):
        status[found & (bmd < scale.lowest)] = BELOW
        status[found & (bmd >= scale.lowest) & (bmd <= scale.top)] = WITHIN
        status[found & (bmd > scale.top)] = ABOVE
    return status


# ---------------------------------------------------------------------------------------
# What the checks of regulatory guidance read
# ---------------------------------------------------------------------------------------


def _residuals_near(
    fits: DoseCurveFits, genes: _GeneFits, bmd: FloatArray
) -> FloatArray:
    """The scaled residual of the dose group nearest each BMD, the lower dose on a
    tie (BMDS's residual of interest); NaN without a BMD."""
    doses = fits.doses
    out = np.full(bmd.size, np.nan)
    ok = np.isfinite(bmd)
    after = np.clip(np.searchsorted(doses, bmd[ok]), 1, doses.size - 1)
    before = after - 1
    nearest = np.where(bmd[ok] - doses[before] <= doses[after] - bmd[ok], before, after)
    out[ok] = genes.scaled_residuals[np.flatnonzero(ok), nearest]
    return out


def _vehicle_sd_ratio(fits: DoseCurveFits, genes: _GeneFits) -> FloatArray:
    """Under constant variance, the curve's SD over the vehicle replicates' sample SD,
    at least 1e-6, as BMDS checks it; NaN otherwise."""
    n_genes = fits.means.shape[0]
    if fits.variance != "constant" or fits.doses[0] != 0.0 or fits.n_samples[0] < 2:
        return np.full(n_genes, np.nan)
    observed = fits.within_group_sd[:, 0]
    sigma = np.sqrt(genes.variance)
    return np.asarray(sigma / np.maximum(observed, 1e-6), dtype=np.float64)


def _constant_variance_test(fits: DoseCurveFits) -> FloatArray:
    """BMDS's test 2: one variance against a variance per dose group, n log(S_w / n)
    - sum_k n_k log(S_k / n_k) on K - 1 degrees of freedom, under constant variance;
    NaN otherwise and where a group has fewer than two samples or no spread."""
    n_genes, n_groups = fits.means.shape
    if fits.variance != "constant" or np.any(fits.n_samples < 2):
        return np.full(n_genes, np.nan)
    sums = fits.within_group_sd**2 * (fits.n_samples - 1.0)
    n = float(fits.n_samples.sum())
    total = sums.sum(axis=1)
    ok = np.all(sums > 0.0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = n * np.log(total / n) - np.sum(
            fits.n_samples * np.log(sums / fits.n_samples), axis=1
        )
    p = np.asarray(
        chi2_sf(np.maximum(statistic, 0.0), n_groups - 1.0), dtype=np.float64
    )
    return np.where(ok, p, np.nan)


# ---------------------------------------------------------------------------------------
# The documents' choices of each gene's model
# ---------------------------------------------------------------------------------------

NTP_2018, ETAP_2024, BMDS_2026 = "ntp_2018", "etap_2024", "bmds_2026"
DOCUMENTS = {NTP_2018: "bmdexpress", ETAP_2024: "bmdexpress", BMDS_2026: "epa"}
"""Each document's choice of model, with the set of the program it runs: BMDS 2's
programs in BMDExpress 2 for NTP 2018 and ETAP, pybmds 25.2 for BMDS's
recommendation."""
BMDS_MODELS = (
    "linear",
    "polynomial_2",
    "power",
    "hill",
    "exponential_3",
    "exponential_5",
)
"""The continuous models that BMDS fits and recommends among."""
HILL_K_FRACTION = 1.0 / 3.0
"""ETAP leaves a Hill fit out of the choice when its k is below this fraction of the
lowest positive dose (``kParameterValue`` 3 in BMDExpress)."""
CLOSE_BMDLS = 3.0
"""BMDS recommends the lowest AIC when the viable models' BMDLs lie within this
factor of each other, and the lowest BMDL otherwise."""
FIT_P = 0.1
# Why each gene has its model.
LOWEST_AIC = "lowest AIC"
HILL_LEFT_OUT = "lowest AIC, Hill left out: k below a third of the lowest dose"
LATER_AIC = "lowest AIC of the models with a BMD and bounds"
NO_CHOICE = "no model with a BMD and bounds"
VIABLE_AIC = "lowest AIC of the viable models, BMDLs within 3-fold"
VIABLE_BMDL = "lowest BMDL of the viable models, BMDLs more than 3-fold apart"
NO_VIABLE = "no viable model"
# Beyond the search's highest dose, the change of every model of these sets has
# reached its limit at this multiple of it: a model whose change reaches the BMR
# there crosses it beyond the search, as BMDS 2 reports such BMDs.
_FAR = 1e6


def _bmds2_counted(fits: DoseCurveFits) -> FloatArray:
    """The mean parameters of each fit that BMDS 2's programs count, in the AIC and
    the degrees of freedom of test 4: those not at a bound that the programs have
    (Hill's and power's n, and exponential 3 and 5's d, at 1 or 18; Hill's k, the
    exponentials' a, b and c at 0). CytePool's own search limits, such as Hill's k at
    1,000 times the top dose in ``"bmdexpress"``, are not bounds of the programs."""
    names = SET_PARAMETERS[fits.model_set]
    out = np.full(fits.aic.shape, np.nan)
    for m, name in enumerate(fits.models):
        model = model_of(fits.model_set, name)
        at = np.zeros(fits.aic.shape[0])
        for parameter in model.parameters:
            value = fits.parameters[:, m, names.index(parameter)]
            if parameter in ("n", "d") and model.kind != POLYNOMIAL:
                at += (value == EXPONENT_MIN) | (value == EXPONENT_MAX)
            elif parameter in ("k", "a", "c") or (
                parameter == "b" and name != "linear"
            ):
                at += value == 0.0
        out[:, m] = model.n_parameters - at
    return out


def _pybmds_counted(fits: DoseCurveFits) -> FloatArray:
    """The mean parameters of each fit that pybmds counts in its AIC and the degrees
    of freedom of its test 4: those not at a bound of the fits' set, which mirror
    pybmds's bounds in ``"epa"``."""
    sizes = np.array([model_of(fits.model_set, m).n_parameters for m in fits.models])
    bounded = np.vectorize(lambda text: len(text.split(", ")) if text else 0)(
        fits.parameters_at_bounds
    )
    return np.asarray(sizes[None, :] - bounded, dtype=np.float64)


def _test_4_p_values(fits: DoseCurveFits, counted: FloatArray) -> FloatArray:
    """BMDS's test 4 of every fit, ``(n_genes, n_models)``, on K minus ``counted``
    degrees of freedom, as the documents' programs count a fit's parameters; NaN
    where they are none, where the groups have no spread and without a fit."""
    n_groups = fits.doses.size
    found = np.full(counted.shape, np.nan)
    for m in range(len(fits.models)):
        fitted = np.isfinite(fits.log_likelihood[:, m])
        if fitted.any():
            q_min = _GeneFits.of(fits, np.where(fitted, m, -1)).q_min
            found[:, m] = np.where(
                fitted, _test_4(fits, q_min, n_groups - counted[:, m]), np.nan
            )
    return found


def _lowest(
    aic: FloatArray, open_: np.ndarray, order: FloatArray | None = None
) -> np.ndarray:
    """Each row's model of lowest AIC among the open ones, a tie within ``TIE``
    going to the lowest ``order`` (fewer parameters) and then to the first in set
    order; -1 where none is open."""
    values = np.where(open_, aic, np.inf)
    lowest = values.min(axis=1)
    tied = open_ & (values <= lowest[:, None] + TIE)
    if order is not None:
        rank = np.where(tied, order, np.inf)
        tied &= rank == rank.min(axis=1, keepdims=True)
    return np.where(open_.any(axis=1), np.argmax(tied, axis=1), -1)


def _per_model(
    fits: DoseCurveFits,
    bmr: BenchmarkResponse,
    m: int,
    with_bounds: bool,
    rows: np.ndarray | None = None,
    piece: bool = False,
) -> tuple[_GeneFits, dict[str, np.ndarray]]:
    """Model m's BMDs of every gene it was fitted to, or of ``rows`` only, with
    bounds when asked, those of the piece that holds the BMD with ``piece``."""
    chosen = np.where(fits.not_fitted_reason[:, m] == "", m, -1).astype(np.int64)
    if rows is not None:
        keep = np.zeros(chosen.size, dtype=bool)
        keep[rows] = True
        chosen = np.where(keep, chosen, -1)
    genes = _GeneFits.of(fits, chosen)
    return genes, _for_response(fits, genes, bmr, with_bounds, piece)


def _reaches_beyond(
    fits: DoseCurveFits, genes: _GeneFits, row: dict[str, np.ndarray]
) -> np.ndarray:
    """Whether each gene's change, not reached within the search, reaches the BMR's
    size beyond it."""
    scale = DoseScale.of(fits.doses)
    far = _FAR * search_range(scale.lowest, scale.top)[1]
    out = np.zeros(genes.chosen.size, dtype=bool)
    for g in np.flatnonzero((row["status"] == NOT_REACHED) & (genes.chosen >= 0)):
        values = mean_values(
            int(genes.kinds[g]), genes.theta[g][None, :], np.array([0.0, far])
        )
        out[g] = abs(values[0, 1] - values[0, 0]) >= row["size"][g]
    return out


def _bmdexpress_choice(
    fits: DoseCurveFits, bmr: BenchmarkResponse, document: str
) -> tuple[np.ndarray, np.ndarray]:
    """NTP's and ETAP's choice as BMDExpress 2 makes it: the lowest AIC as BMDS 2
    counts it, among the models whose BMD is reached (within the search or beyond
    it) and whose BMDL and BMDU are numbers, 0 and infinity included; ETAP leaves out
    a Hill fit whose k is below a third of the lowest dose. Bounds are computed for
    each gene's best remaining model, round by round."""
    n_genes, n_models = fits.aic.shape
    aic = -2.0 * fits.log_likelihood + 2.0 * (_bmds2_counted(fits) + 1.0)
    fitted = (fits.not_fitted_reason == "") & np.isfinite(aic)
    reached = np.zeros((n_genes, n_models), dtype=bool)
    for m in range(n_models):
        genes, row = _per_model(fits, bmr, m, False)
        reached[:, m] = np.isfinite(row["bmd"]) | _reaches_beyond(fits, genes, row)
    flagged = (
        _flagged_hill(fits)
        if document == ETAP_2024
        else np.zeros((n_genes, n_models), dtype=bool)
    )
    open_ = fitted & reached & ~flagged

    def with_bounds(rows: np.ndarray, models: np.ndarray) -> np.ndarray:
        ok = np.zeros(rows.size, dtype=bool)
        for m in np.unique(models):
            mine = models == m
            _, row = _per_model(fits, bmr, int(m), True, rows[mine], piece=True)
            bmdl, bmdu = row["bmdl"][rows[mine]], row["bmdu"][rows[mine]]
            beyond = row["status"][rows[mine]] == NOT_REACHED
            ok[mine] = beyond | (~np.isnan(bmdl) & ~np.isnan(bmdu))
        return ok

    chosen = _first_with_bounds(aic, open_, with_bounds)
    first = _lowest(aic, fitted)
    without_hill = _lowest(aic, fitted & ~flagged)
    why = np.full(n_genes, NO_CHOICE, dtype=object)
    picked = chosen >= 0
    why[picked] = LATER_AIC
    why[picked & (chosen == without_hill) & (without_hill != first)] = HILL_LEFT_OUT
    why[picked & (chosen == first)] = LOWEST_AIC
    return chosen, why


def _check_inputs(
    fits: DoseCurveFits,
    genes: _GeneFits,
    row: dict[str, np.ndarray],
    fit_p_value: FloatArray,
    with_bounds: bool,
) -> FlagInputs:
    """What the documents' checks read of one model's BMDs, with ``fit_p_value`` as
    its fit test."""
    return FlagInputs(
        bmd=row["bmd"],
        bmdl=row["bmdl"],
        bmdu=row["bmdu"],
        bounds=with_bounds,
        lowest_dose=float(fits.doses[fits.doses > 0.0][0]),
        top_dose=float(fits.doses[-1]),
        fit_p_value=fit_p_value,
        residual_near_bmd=_residuals_near(fits, genes, row["bmd"]),
        residual_at_vehicle=genes.scaled_residuals[:, 0],
        vehicle_sd_ratio=_vehicle_sd_ratio(fits, genes),
        constant_variance_p_value=_constant_variance_test(fits),
    )


def _bins(texts: list[str]) -> np.ndarray:
    """BMDS's bin of each row from the checks it fails: 2 unusable, 1 questionable,
    0 viable (a warning keeps a model viable)."""
    return np.array(
        [
            2 if "(unusable)" in text else 1 if "(questionable)" in text else 0
            for text in texts
        ],
        dtype=np.int64,
    )


def _bmds_choice(
    fits: DoseCurveFits, bmr: BenchmarkResponse
) -> tuple[np.ndarray, np.ndarray]:
    """BMDS's recommended model as pybmds 25.2 makes it: each of its six continuous
    models binned by the default checks with pybmds's test 4 as the fit test (its
    degrees of freedom leave out the parameters at a bound, as its AIC does), bounds
    computed only for the models that the checks without bounds leave viable; among
    the viable, the lowest AIC (parameters at a bound left out) when their BMDLs lie
    within 3-fold of each other, the lowest BMDL otherwise; ties to the fewest
    parameters, then the set's order."""
    n_genes, n_models = fits.aic.shape
    positions = [fits.models.index(m) for m in BMDS_MODELS if m in fits.models]
    sizes = np.array([model_of(fits.model_set, m).n_parameters for m in fits.models])
    counted = _pybmds_counted(fits)
    aic = -2.0 * fits.log_likelihood + 2.0 * (counted + 1.0)
    test_4 = _test_4_p_values(fits, counted)
    viable = np.zeros((n_genes, n_models), dtype=bool)
    bmdl = np.full((n_genes, n_models), np.nan)
    for m in positions:
        genes, row = _per_model(fits, bmr, m, False)
        fitted = genes.chosen >= 0
        keep = fitted & (
            _bins(
                failed_checks(
                    BMDS_2026, _check_inputs(fits, genes, row, test_4[:, m], False)
                )
            )
            == 0
        )
        if keep.any():
            genes, row = _per_model(fits, bmr, m, True, np.flatnonzero(keep))
            texts = failed_checks(
                BMDS_2026, _check_inputs(fits, genes, row, test_4[:, m], True)
            )
            keep &= _bins(texts) == 0
            bmdl[:, m] = np.where(keep, row["bmdl"], np.nan)
        viable[:, m] = keep
    return _recommended(aic, bmdl, viable, sizes + 1.0)


def _recommended(
    aic: FloatArray, bmdl: FloatArray, viable: np.ndarray, n_parameters: FloatArray
) -> tuple[np.ndarray, np.ndarray]:
    """BMDS's recommendation among each gene's viable models: the lowest AIC when the
    largest of their BMDLs above 0 is at most 3 times the smallest, the lowest BMDL
    otherwise; a tie to the fewest parameters (``n_parameters``, the variance's
    included), then the first in set order. -1 and the reason where none is
    viable."""
    with np.errstate(invalid="ignore", divide="ignore"):
        positive = np.where(viable & (bmdl > 0.0), bmdl, np.nan)
        ratio = np.nanmax(positive, axis=1, initial=-np.inf) / np.nanmin(
            positive, axis=1, initial=np.inf
        )
    by_aic = ratio <= CLOSE_BMDLS
    values = np.where(by_aic[:, None], aic, bmdl)
    order = np.broadcast_to(n_parameters, viable.shape).astype(np.float64)
    chosen = _lowest(values, viable, order)
    why = np.full(viable.shape[0], NO_VIABLE, dtype=object)
    why[(chosen >= 0) & by_aic] = VIABLE_AIC
    why[(chosen >= 0) & ~by_aic] = VIABLE_BMDL
    return chosen, why


def _checked_document(fits: DoseCurveFits, model: Any, with_bounds: bool) -> str | None:
    """The document whose choice ``model`` names, checked against the fits; None for
    ``"best"`` and the models' names."""
    if not isinstance(model, str) or model not in DOCUMENTS:
        return None
    needed = DOCUMENTS[model]
    if fits.model_set != needed:
        program = (
            "pybmds 25.2" if model == BMDS_2026 else "BMDS 2's programs in BMDExpress 2"
        )
        raise ValueError(
            f"model={model!r} is the choice that {program} makes among the models of "
            f"the set {needed!r}, and these fits are of {fits.model_set!r}; fit the "
            f"curves with models={needed!r}"
        )
    if fits.variance != "constant":
        raise ValueError(
            f"model={model!r} reads each model's likelihood of the samples, its test 4 "
            f"and AIC, which the {fits.variance} fits do not have; fit the curves under "
            "constant variance"
        )
    if not with_bounds:
        raise ValueError(
            f"model={model!r} keeps a model only where its BMD has bounds; leave bounds "
            "at True, which needs replicates within dose groups"
        )
    return model


def _vehicle_sd_bmds(
    fits: DoseCurveFits, genes: _GeneFits, bmr: BenchmarkResponse
) -> FloatArray:
    """Each gene's BMD at the BMR's multiple of the vehicle replicates' sample SD,
    ETAP's wording of its BMR, without bounds; NaN for an absolute BMR, without a
    model, or where the vehicle has fewer than two samples or no spread."""
    n_genes = genes.chosen.size
    if bmr.kind == ABSOLUTE or fits.doses[0] != 0.0 or fits.n_samples[0] < 2:
        return np.full(n_genes, np.nan)
    sizes = bmr.value * fits.within_group_sd[:, 0]
    return _crossings(fits, genes, sizes)


def _crossings(fits: DoseCurveFits, genes: _GeneFits, sizes: FloatArray) -> FloatArray:
    """The BMD of each gene's model at its own size: the first dose in the search at
    which the change reaches it; NaN where it is not reached, without a model, or
    for a size that is not a number above 0."""
    scale = DoseScale.of(fits.doses)
    low, high = search_range(scale.lowest, scale.top)
    bmd = np.full(genes.chosen.size, np.nan)
    defined = np.isfinite(sizes) & (sizes > 0.0)
    for kind in np.unique(genes.kinds[genes.kinds >= 0]):
        rows = np.flatnonzero((genes.kinds == kind) & defined)
        if rows.size == 0:
            continue
        theta = np.ascontiguousarray(np.stack([genes.theta[g] for g in rows]))
        turns = np.ascontiguousarray(turn_doses(int(kind), theta, high))
        found = np.full(rows.size, np.nan)
        signs = np.zeros(rows.size)
        codes = np.zeros(rows.size, dtype=np.int64)
        first_crossings(
            int(kind),
            theta,
            np.ascontiguousarray(sizes[rows]),
            turns,
            low,
            high,
            found,
            signs,
            codes,
        )
        bmd[rows] = np.where(codes == NEVER_CROSSED, np.nan, found)
    return bmd


def _flagged_hill(fits: DoseCurveFits) -> np.ndarray:
    """Where a Hill fit's k lies below a third of the lowest positive dose, which
    BMDExpress flags (``kParameterValue`` 3) and ETAP leaves out of its choice."""
    flagged = np.zeros(fits.aic.shape, dtype=bool)
    if "hill" in fits.models:
        m = fits.models.index("hill")
        k = fits.parameters[:, m, SET_PARAMETERS[fits.model_set].index("k")]
        lowest = float(fits.doses[fits.doses > 0.0][0])
        with np.errstate(invalid="ignore"):
            flagged[:, m] = k < HILL_K_FRACTION * lowest
    return flagged


def _first_with_bounds(
    aic: FloatArray,
    open_: np.ndarray,
    with_bounds: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Each gene's model of lowest AIC among the open ones whose bounds are numbers:
    ``with_bounds(genes, models)`` says so for each gene's best remaining model, round
    by round, and a model without them leaves the race; -1 where none is left."""
    remaining = open_.copy()
    chosen = np.full(aic.shape[0], -1, dtype=np.int64)
    undecided = remaining.any(axis=1)
    while undecided.any():
        rows = np.flatnonzero(undecided)
        best = _lowest(aic[rows], remaining[rows])
        ok = with_bounds(rows, best)
        chosen[rows[ok]] = best[ok]
        undecided[rows[ok]] = False
        remaining[rows[~ok], best[~ok]] = False
        undecided[rows] &= remaining[rows].any(axis=1)
    return chosen
