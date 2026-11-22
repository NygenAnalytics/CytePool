"""NTP's and ETAP's rule sets: each setting with its value, BMDExpress's setting and its
source, as data; then the steps of the documents' workflow that run on arrays.

NTP 2018 and ETAP (US EPA 2024) run BMDExpress 2, whose settings the documents leave
partly open. Where NTP 2018 leaves a setting to each platform or does not state it,
the NTP rule set takes ETAP's, since ETAP states that its steps are consistent with
NTP 2018 (Sec. 3.4.5.3); neither document states the number of permutations of
Williams' trend test. :func:`~cytepool.dose_response` reads the data below for its
``rules`` option, :func:`~cytepool.check_dose_design` reads ETAP's design minimums,
and the documentation prints both rule sets from them.

References:
    NTP (2018). NTP research report on National Toxicology Program approach to
    genomic dose-response modeling. NTP Research Report 5. National Institute of
    Environmental Health Sciences. doi:10.22427/ntp-rr-5

    US EPA (2024). Standard Methods for Development of EPA Transcriptomic Assessment
    Products (ETAPs). EPA/600/X-23/083, US Environmental Protection Agency, Office of
    Research and Development. doi:10.23645/epacomptox.25365496

    Thomas RS, Allen BC, Nong A, Yang L, Bermudez E, Clewell HJ, Andersen ME (2007). A
    method to integrate benchmark dose estimates with genomic data to assess the
    functional effects of chemical exposure. Toxicological Sciences 98, 240-248.
    doi:10.1093/toxsci/kfm092

    Ringblom J, Johanson G, Öberg M (2014). Current modeling practice may lead to
    falsely high benchmark dose estimates. Regulatory Toxicology and Pharmacology 69,
    171-177. doi:10.1016/j.yrtph.2014.03.004
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .normalization import cpm
from .stats.distributions import f_sf
from .stats.multiple_testing import adjust_p_values

FloatArray = npt.NDArray[np.float64]

NTP_2018, ETAP_2024 = "ntp_2018", "etap_2024"
RULE_SETS = (NTP_2018, ETAP_2024)

# ETAP's design minimums: a vehicle and at least 8 doses, spaced by half a log10 down
# from the highest, the lowest a full log10 below the second lowest (Sec. 3.4.1.3); 4
# animals of each sex per dose group (Sec. 3.4.2); 2 passing samples at each dose level
# and 3 passing vehicle samples (Sec. 3.4.5.2). A step's log10 is read to one decimal
# place, the precision of "half-log10".
ETAP_DOSES = 8
ETAP_STEP, ETAP_LOWEST_STEP, ETAP_STEP_TOLERANCE = 0.5, 1.0, 0.05
ETAP_ANIMALS = 4
ETAP_PASSING_SAMPLES = 2
ETAP_VEHICLE_SAMPLES = 3

# The steps both rule sets share (US EPA 2024 Sec. 3.4.5.2 and 3.4.5.3).
MEAN_COUNT = 5.0
"""Genes whose mean count over the cell type's samples is below this are left out."""
ANOVA_FDR = 0.05
"""A cell type is analysed when a gene's adjusted ANOVA p-value is below this."""
WILLIAMS_P = 0.05
"""A gene passes Williams' trend test when its p-value is below this, unadjusted."""
FOLD_CHANGE = 1.5
"""A gene passes the fold change when a dose's mean differs from the vehicle's by at
least this factor, either way."""
N_PERMUTATIONS = 10_000
"""The permutations of Williams' trend test: neither document states a number; at this
many, a p-value near 0.05 has a Monte Carlo SD of about 0.003."""


@dataclass(frozen=True)
class Setting:
    """One setting of a rule set: what it is, its value in plain words, the setting of
    BMDExpress 2.30.520 that gives it (empty where BMDExpress has none), and where it
    comes from."""

    setting: str
    value: str
    bmdexpress: str
    source: str


@dataclass(frozen=True)
class RuleSet:
    """A document's workflow: its benchmark response in SDs, the fraction of the
    lowest dose below which a Hill fit's k leaves it out of the choice (None for no
    such rule), and its settings in workflow order, then the steps it names that
    CytePool does not apply."""

    name: str
    document: str
    bmr_sd: float
    hill_k_fraction: float | None
    settings: tuple[Setting, ...]
    not_applied: tuple[Setting, ...]


_ETAP = "US EPA 2024"
_NTP = "NTP 2018"
_FROM_ETAP = "taken from ETAP (US EPA 2024, Sec. 3.4.5.3)"
_PREFILTER = "williams prefilter"

_SHARED_NOT_APPLIED = (
    Setting(
        "one gene per feature",
        "not applied: a cube's features are taken as genes; map probes to genes and "
        "keep one feature per gene before pooling",
        "categoryAnalysisConfigs: removePromiscuousProbes",
        f"{_NTP}, p. 9; {_ETAP}, Sec. 3.4.5.4",
    ),
    Setting(
        "models without a spare dose group",
        "not fitted: a model of p mean parameters needs K - p >= 1 dose groups, "
        "where BMDExpress fits it",
        "",
        "Ringblom et al. 2014",
    ),
    Setting(
        "calibration of the prefilter",
        "not applied: NTP sets the prefilter's thresholds for each platform on null "
        "data sets of vehicle controls, and none exist for pseudobulk profiles; "
        "permutation_check checks the user's design",
        "",
        f"{_NTP}, p. 5",
    ),
)

ETAP = RuleSet(
    name=ETAP_2024,
    document=_ETAP,
    bmr_sd=1.349,
    hill_k_fraction=1.0 / 3.0,
    settings=(
        Setting(
            "data set",
            "one chemical, sex and tissue: the samples of one call in one cell type",
            "expressionDataConfigs: one file per data set",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "design minimums",
            f"{ETAP_DOSES} doses and a vehicle, spaced by half a log10 with the lowest "
            f"step a full log10; {ETAP_ANIMALS} animals per dose group; a dose level "
            f"with fewer than {ETAP_PASSING_SAMPLES} passing samples is left out, and "
            f"a cell type with fewer than {ETAP_VEHICLE_SAMPLES} passing vehicle "
            "samples is not modelled",
            "",
            f"{_ETAP}, Sec. 3.4.1.3, 3.4.2 and 3.4.5.2",
        ),
        Setting(
            "values",
            f"genes with a mean count of at least {MEAN_COUNT:g}, counts per million "
            "over them, log2(CPM + 1)",
            "logTransformation: BASE2",
            f"{_ETAP}, Sec. 3.4.5.2",
        ),
        Setting(
            "ANOVA pre-modelling test",
            "one-way ANOVA of the dose groups; the cell type is analysed when a gene's "
            f"Benjamini-Hochberg adjusted p-value is below {ANOVA_FDR:g}, and is "
            "inactive otherwise",
            f"anova prefilter: pValueCutoff {ANOVA_FDR:g}, "
            "useMultipleTestingCorrection true",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "Williams' trend test",
            f"two-sided, {N_PERMUTATIONS:,} permutations; p < {WILLIAMS_P:g}, not "
            "adjusted",
            f"{_PREFILTER}: numberOfPermutations {N_PERMUTATIONS}, pValueCutoff "
            f"{WILLIAMS_P:g}, useMultipleTestingCorrection false",
            f"{_ETAP}, Sec. 3.4.5.3; permutations not stated",
        ),
        Setting(
            "fold change",
            f"at least {FOLD_CHANGE:g}-fold from the vehicle's mean at one dose or "
            "more, either way",
            f"{_PREFILTER}: useFoldChange true, foldChange {FOLD_CHANGE:g}",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "models",
            "linear, polynomial 2 (unrestricted), power, Hill and exponential 2 to 5, "
            "with the bounds of BMDS 2's programs",
            "modelConfigs: poly 1, poly 2, power, hill, exp 2 to 5",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "variance",
            "constant",
            "constantVariance true",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "power exponent",
            "at least 1, at most 18",
            "restrictPower true",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "benchmark response",
            "1.349 SD, the SD of the fitted curve as BMDExpress estimates it; the "
            "BMD at 1.349 times the vehicle replicates' SD, ETAP's wording, prints "
            "beside it",
            "bmrFactor 1.349",
            f"{_ETAP}, Sec. 3.4.5.3; Thomas et al. 2007",
        ),
        Setting(
            "model choice",
            "the lowest AIC, counting the parameters not at a bound, among the "
            "models with a BMD, BMDL and BMDU, after leaving out a Hill fit whose k "
            "is below a third of the lowest dose",
            "bestPolyTest 2; flagHillWithKParameter true, kParameterValue 3, "
            "bestModelSelectionWithFlaggedHill 2",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "fit test",
            "BMDS's test 4: the likelihood ratio of the curve against a mean for "
            "each dose group",
            "fitPValue",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
        Setting(
            "acceptance",
            "BMD below the highest dose, fit p-value above 0.1 and BMD / BMDL below 20",
            "categoryAnalysisConfigs: removeBMDGreaterHighDose true, bmdPValueCutoff "
            "0.1, bmdBMDLRatioMin 20",
            f"{_ETAP}, Sec. 3.4.5.3",
        ),
    ),
    not_applied=(
        Setting(
            "sample quality",
            "not applied: check the cells and counts of each sample when pooling "
            "(min_cells, min_counts)",
            "",
            f"{_ETAP}, Sec. 3.4.5.1",
        ),
        Setting(
            "outlier samples",
            "not applied: leave outlier samples out of the design first",
            "",
            f"{_ETAP}, Sec. 3.4.5.2",
        ),
        *_SHARED_NOT_APPLIED[:2],
    ),
)

NTP = RuleSet(
    name=NTP_2018,
    document=_NTP,
    bmr_sd=1.0,
    hill_k_fraction=None,
    settings=(
        Setting(
            "data set",
            "one chemical: the samples of one call in one cell type",
            "expressionDataConfigs: one file per data set",
            f"{_NTP}, p. 5",
        ),
        Setting(
            "design minimums",
            "none stated",
            "",
            f"{_NTP}, pp. 10 to 11",
        ),
        Setting(
            "values",
            f"genes with a mean count of at least {MEAN_COUNT:g}, counts per million "
            "over them, log2(CPM + 1)",
            "logTransformation: BASE2",
            f"{_NTP}, p. 9 (log2 values after normalization); {_FROM_ETAP}",
        ),
        Setting(
            "ANOVA pre-modelling test",
            "one-way ANOVA of the dose groups; the cell type is analysed when a gene's "
            f"Benjamini-Hochberg adjusted p-value is below {ANOVA_FDR:g}",
            f"anova prefilter: pValueCutoff {ANOVA_FDR:g}, "
            "useMultipleTestingCorrection true",
            f"{_NTP}, p. 5",
        ),
        Setting(
            "Williams' trend test",
            f"two-sided, {N_PERMUTATIONS:,} permutations; p < {WILLIAMS_P:g}, not "
            "adjusted",
            f"{_PREFILTER}: numberOfPermutations {N_PERMUTATIONS}, pValueCutoff "
            f"{WILLIAMS_P:g}, useMultipleTestingCorrection false",
            f"{_NTP}, pp. 5 to 6 (threshold set per platform); {_FROM_ETAP}; "
            "permutations not stated",
        ),
        Setting(
            "fold change",
            f"at least {FOLD_CHANGE:g}-fold from the vehicle's mean at one dose or "
            "more, either way",
            f"{_PREFILTER}: useFoldChange true, foldChange {FOLD_CHANGE:g}",
            f"{_NTP}, pp. 5 to 6 (set per platform); {_FROM_ETAP}",
        ),
        Setting(
            "models",
            "linear, polynomial 2 (unrestricted), power, Hill and exponential 2 to 5, "
            "with the bounds of BMDS 2's programs",
            "modelConfigs: poly 1, poly 2, power, hill, exp 2 to 5",
            f"{_NTP}, p. 6",
        ),
        Setting(
            "variance",
            "constant",
            "constantVariance true",
            f"{_NTP}, p. 7",
        ),
        Setting(
            "power exponent",
            "at least 1, at most 18; NTP's equation allows exponents above 0",
            "restrictPower true",
            f"{_NTP}, p. 7; BMDExpress's default",
        ),
        Setting(
            "benchmark response",
            "1 SD, the SD of the fitted curve under constant variance",
            "bmrFactor 1",
            f"{_NTP}, p. 7",
        ),
        Setting(
            "model choice",
            "the lowest AIC, counting the parameters not at a bound, among the "
            "models with a BMD, BMDL and BMDU",
            "bestPolyTest 2; flagHillWithKParameter false",
            f"{_NTP}, p. 8",
        ),
        Setting(
            "fit test",
            "BMDS's test 4: the likelihood ratio of the curve against a mean for "
            "each dose group",
            "fitPValue",
            f"{_NTP}, p. 9",
        ),
        Setting(
            "acceptance",
            "BMD, BMDL and BMDU computed, BMD below the highest dose, fit p-value "
            "above 0.1 and BMDU / BMDL below 40; a BMD more than 10-fold below the "
            "lowest dose is flagged and kept",
            "categoryAnalysisConfigs: removeBMDGreaterHighDose true, bmdPValueCutoff "
            "0.1, bmduBMDLRatioMin 40",
            f"{_NTP}, pp. 8 to 9",
        ),
    ),
    not_applied=_SHARED_NOT_APPLIED,
)

BY_NAME = {NTP_2018: NTP, ETAP_2024: ETAP}


# The steps that run on arrays.


def values_by_etap(counts: FloatArray) -> tuple[npt.NDArray[np.bool_], FloatArray]:
    """The genes kept and their values, as ETAP prepares counts (Sec. 3.4.5.2): genes
    whose mean count over the samples is at least ``MEAN_COUNT``, counts per million
    over the genes kept, log2(CPM + 1).

    Args:
        counts: ``(n_samples, n_genes)`` counts of one cell type's samples.

    Returns:
        Whether each gene is kept, and the ``(n_samples, n_kept)`` values.
    """
    kept = counts.mean(axis=0) >= MEAN_COUNT
    return kept, np.log2(cpm(counts[:, kept]) + 1.0)


def anova_premodelling_test(
    values: FloatArray, labels: npt.NDArray[np.intp], n_groups: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Each gene's one-way ANOVA of the dose groups: F, its p-value on K - 1 and N - K
    degrees of freedom and the Benjamini-Hochberg adjusted p-values over the genes.
    A gene whose values do not vary gets NaN and is left out of the adjustment; one
    whose groups have no spread within them gets an infinite F and p-value 0.

    Args:
        values: ``(n_samples, n_genes)`` values.
        labels: The dose group of each sample, 0 to K - 1.
        n_groups: K.
    """
    n = values.shape[0]
    sizes = np.bincount(labels, minlength=n_groups).astype(np.float64)
    centred = values - values.mean(axis=0)
    one_hot = (labels[None, :] == np.arange(n_groups)[:, None]).astype(np.float64)
    sums = one_hot @ centred
    between = (sums**2 / sizes[:, None]).sum(axis=0)
    total = (centred**2).sum(axis=0)
    within = np.maximum(total - between, 0.0)
    constant = np.ptp(values, axis=0) == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (between / (n_groups - 1)) / (within / (n - n_groups))
    f = np.where(constant, np.nan, f)
    p = np.where(constant, np.nan, f_sf(f, n_groups - 1, n - n_groups))
    return f, p, adjust_p_values(p)


def largest_log2_fold_changes(
    values: FloatArray, labels: npt.NDArray[np.intp], n_groups: int
) -> FloatArray:
    """Each gene's change from the vehicle's mean (group 0) at the dose whose change
    is largest either way, the lower dose on a tie, in the values' log2 units."""
    sizes = np.bincount(labels, minlength=n_groups).astype(np.float64)
    one_hot = (labels[None, :] == np.arange(n_groups)[:, None]).astype(np.float64)
    means = (one_hot @ values) / sizes[:, None]
    changes = means[1:] - means[0]
    largest = np.argmax(np.abs(changes), axis=0)
    return changes[largest, np.arange(values.shape[1])]


def passes_prefilter(p_values: FloatArray, log2_changes: FloatArray) -> np.ndarray:
    """Williams' p-value below ``WILLIAMS_P`` and a change of at least ``FOLD_CHANGE``
    fold either way; NaN passes neither."""
    with np.errstate(invalid="ignore"):
        return np.asarray(
            (p_values < WILLIAMS_P) & (np.abs(log2_changes) >= np.log2(FOLD_CHANGE))
        )
