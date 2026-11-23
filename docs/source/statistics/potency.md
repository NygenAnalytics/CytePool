---
description: Benchmark doses of fitted dose-response curves with one-sided 95% profile-likelihood bounds, where each lies against the doses, the checks of guidance, and NTP's and ETAP's workflows.
---

# Potency and points of departure

[Dose response](dose_response.md) fitted curves of named models to each gene's dose groups. A curve says what shape a response takes; a toxicologist also asks at which dose the response reaches a size that matters, and reads that dose as a benchmark dose (BMD): the dose at which the curve's change from its level at dose 0 reaches a stated benchmark response (BMR), with a lower bound (BMDL) and an upper bound (BMDU) [@crump1984; @usepa2012]. A pharmacologist's measure of potency, the EC50, is the dose of half a curve's full change, such as the Hill model's k in the table of curves; a BMD is the dose of a stated change, and EC50s of every model, with intervals, are not implemented yet. {api}`dose_response` with `bmr` computes BMDs for every gene with a curve in each cell type, and {api}`benchmark_doses` for curves fitted outside a cube; [Dose response per cell type](../guides/dose_response.ipynb) runs both on a simulated series. The lowest dose that a test calls against the vehicle answers a neighbouring question, at an error rate and without a curve ([Choosing a method](choosing_a_method.md#from-which-dose)).

(benchmark-doses)=
## Benchmark doses

The BMD of a curve f is the lowest dose x above 0 at which |f(x) - f(0)| = R, R the BMR's size. Regulatory guidance states R in two ways for genes, and {api}`BenchmarkResponse` holds either:

| BMR | R | Guidance |
| --- | --- | --- |
| `BenchmarkResponse.sd(multiple)` | the multiple times the SD of a vehicle replicate | 1 SD: the default of EPA's Benchmark Dose Software (BMDS) and of the National Toxicology Program (NTP 2018) [@ntp2018]; 1.349 SD: EPA's Transcriptomic Assessment Products (ETAP) [@usepa2024], after @thomas2007, because a gene's adverse direction is unknown |
| `BenchmarkResponse.absolute(change)` | the change, in the values' units | on log2 values, `absolute(np.log2(1.5))` is a 1.5-fold change and `absolute(np.log2(1.1))` a 1.1-fold change, up or down |

On the series of [Dose response](dose_response.md#dose-response-tests):

```python
import numpy as np

import cytepool as cp

cube, truth = cp.simulate_dose_series(
    [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    replicates=3,
    cell_types=["hepatocyte", "stellate", "cholangiocyte"],
    cells=[300, 60, 6],
    composition={"stellate": [1, 1, 1, 1, 1, 1, 1, 1, 0.1]},
    rng=7,
)
axis = cp.DoseAxis("dose", unit="ug/kg", metric="administered")
bmr = [cp.BenchmarkResponse.sd(1.349), cp.BenchmarkResponse.absolute(np.log2(1.5))]
result = cp.dose_response(
    cube,
    cp.Design(cube.samples, "~ 1"),
    dose=axis,
    models="epa",
    bmr=bmr,
    flags="etap_2024",
)
bmds = result.table("hepatocyte", "benchmark_doses")
```

The printout gains a line per cell type of where the BMDs of each BMR lie, such as `BMDs at 1.349 SD: 149 within the doses, 4 below the lowest dose, 2 below a hundredth of the lowest dose, 598 above the top dose, 64 not reached` for the hepatocytes' 817 genes. Most genes do not respond, and the slight slopes of their best curves reach the BMR only beyond the top dose, or never; read the BMDs of the genes that pass the gate, whose call `passes_gate` gives beside each BMD.

The SD of an SD-based BMR is that of the curve's likelihood, as BMDS estimates it with the curve: under constant variance the curve's maximum-likelihood SD, sqrt((S_w + Q) / n), with S_w the within-group sum of squares, Q the curve's weighted sum of squares about the dose-group means and n the samples. Where the curves weigh the dose-group means of a moderated fit, the likelihood holds the fit's variance s², on df degrees of freedom, with the curve, so the SD is its maximum-likelihood value sqrt((df s² + Q) / (df + K)) at K dose groups, scaled to one vehicle replicate by sqrt(n_0 C_00), with C_00 the unscaled variance of the vehicle's mean over its n_0 samples. `bmr_change` gives each row's R in the values' units, so that an SD-based BMR can be read as a change.

Whatever BMRs are given, the 1 SD BMD and its bounds are added last unless given, since @usepa2012 asks that they "always be presented for comparison purposes".

A curve that rises and then falls, a Gauss-probit bell, can reach the BMR twice, and the BMD is the first crossing, as in DRomics [@delignettemuller2023]: before the turn, or beyond it in the other direction when the curve falls back past its level at dose 0 by R. The search runs from a hundredth of the lowest positive dose, DRomics' `minBMD`, to 50 times the top dose, and `bmd_status` says where the BMD lies:

| `bmd_status` | `bmd` |
| --- | --- |
| `"within the doses"` | from the lowest positive dose to the top dose |
| `"below the lowest dose"` | from a hundredth of the lowest positive dose up to it |
| `"below a hundredth of the lowest dose"` | a crossing below the search, given as a hundredth of the lowest positive dose, as DRomics gives it |
| `"above the top dose"` | up to 50 times the top dose, as BMDS reports BMDs beyond the doses |
| `"not reached"` | NaN: the curve never reaches the BMR up to 50 times the top dose |
| `"no curve"` | NaN: no model to read, none fitted to the gene or none with a finite AICc |
| `"BMR undefined"` | NaN: an SD of 0, values without spread about the curve, or no vehicle group to scale the fit's SD to |

(benchmark-dose-bounds)=
## Bounds by profile likelihood

The BMDL and BMDU are the lowest and highest doses b whose profile log-likelihood lies within half of 2.7055, the chi-square(1) quantile at 0.90, of the curve's maximum: twice the drop, the deviance, lies within 2.7055. The profile log-likelihood at b is the largest log-likelihood of a curve of the same model whose change at b is R, in the direction of the change at the BMD, as BMDS and @usepa2012 (Sec. 2.3.8) define the BMDL. Each bound is one-sided 95%, given the gene's chosen model, and BMDL to BMDU is a two-sided 90% interval, as BMDS and the European Food Safety Authority (EFSA) [@efsa2022] state it. On a developmental toxicity data set, profile-likelihood bounds were similar to bootstrap bounds at a fraction of the cost, while delta-method bounds were usually narrower and appeared unreliable for nonlinear models [@moerbeek2004].

- **The SD inside the profile.** For an SD-based BMR, R moves with the SD, so the profile maximises over the SD too, as BMDS does. A bound that held the SD at its estimate would ignore its error: on straight lines of 9 dose groups x 3, BMDLs at 1.349 times the vehicle replicates' sample SD, held fixed, covered the true BMD in 65% and 78% of datasets at true BMDs of 1 and 10, against 97% with the SD inside the profile. The BMDU covers less, in 89.5% and 91.4% of the same datasets, against 94.7% and 94.1% at a change of 0.4, since the maximum-likelihood SD, with n in its denominator, is biased low, and the BMR's size with it; BMDL to BMDU covered 86.8% and 88.4%, and 88.7% and 88.3% at a change of 0.4, below the nominal 90% ([Validation](../validation.md#validation-dose-response-calibration)).
- **Weighted means.** Where the curves weigh the means of a moderated fit with variance s² on df degrees of freedom, an absolute BMR's deviance is the increase of the weighted sum of squares over s², against F(1, df) at 0.90: for a straight line this is Fieller's bound, exact at any df [@fieller1954]. An SD-based BMR's likelihood adds the scaled chi-square likelihood of s², so that its deviance reads chi-square(1); with infinite df both are the chi-square threshold.
- **The whole region.** The bounds are the ends of the region of doses within the threshold, which can have two pieces: in simulated Hill curves with a weak response, a step just above dose 0 fitted almost as well as the curve, and the region below the BMD reached doses thousands of times lower than the piece around the BMD. A BMDL of 0 says that the region reaches a hundredth of the lowest dose, where the search stops, and a BMDU of infinity that it reaches 50 times the top dose. For a BMD within the doses, the BMDL is at most the top dose, so a BMDU of infinity fails NTP's BMDU/BMDL < 40 and raises EFSA's BMDU/BMDL > 50 indicator whatever the bound's value beyond the search; for a BMD above the top dose, these verdicts can depend on where the search ends. Under NTP's and ETAP's choices of model, the bounds end at the piece that holds the BMD, as BMDS 2's do in BMDExpress ([NTP's and ETAP's rule sets](#rule-sets)).
- **A fit that is not the maximum.** The bounds are measured from the fit. Where the profile finds a curve of larger likelihood than the fit's, at a BMD or at any dose of the search, as a fit that did not converge can leave, the gene gets no bounds at any BMR, and the printout counts its BMDs. Values without spread about the curve, such as an exact line, give an absolute BMR's BMD without bounds.
- **How the profile is found.** At each of 8 doses per decade over the search's range, the curve's linear parameters and the SD are solved in closed form at each point of the curve's grid of nonlinear parameters, and the nonlinear parameters are refined, by a quasi-Newton method with the profile's exact gradient, from the grid's lowest minima and from the optima at the neighbouring dose, outward from the BMD, where the fit itself is the optimum. A parameter that reaches its bound is held there while the gradient points outward. Brent's method finds each bound between two doses of the scan. The search is deterministic and does not depend on the threads.

(benchmark-dose-table)=
## Read the table

`table(group, "benchmark_doses")` of {api}`DoseResponseResult`, and {api}`BenchmarkDoses.table` of {api}`DoseResponseResult.benchmark_doses` or {api}`benchmark_doses`, have one row per BMR and gene:

| Column | What it holds |
| --- | --- |
| `feature_id`, `symbol` | The gene, and its symbol when the cube has one |
| `passes_gate` | The gate's call of the gene, in `table(group, "benchmark_doses")` of a cell type where the gate ran ([Which genes count as responding](dose_response.md#dose-response-gate)) |
| `bmr` | The BMR's label, such as `1.349 SD` or `change of 0.584963` |
| `model` | The gene's model of lowest AICc, or the model named with `model=` |
| `direction` | `up` or `down`, the change at the BMD |
| `bmr_change` | The BMR's size R in the values' units |
| `bmd`, `bmdl`, `bmdu` | The BMD and its bounds |
| `bmd_status` | Where the BMD lies against the doses |
| `flags_<set>` | The checks of each set asked for that the row fails |

Read the BMD with its bounds: a wide interval says that the doses do not pin the BMD down, and NTP leaves out a gene whose BMDU/BMDL is 40 or more; a BMDL of 0 or a BMDU of infinity marks a bound beyond the search. Compare genes on rows of the same BMR. {api}`BenchmarkDoses` keeps the numbers as arrays of one row per BMR, with the statistics that the checks read and the genes' ids, so `result.benchmark_doses("hepatocyte").table(flags="ntp_2018")` gives NTP's checks without computing the bounds again. The result's `settings` record each BMR given as {api}`BenchmarkResponse.to_dict` writes it, and `BenchmarkResponse(**record)` rebuilds it.

(benchmark-dose-flags)=
## Flag sets

`flags` adds, for each named set, the checks of that guidance that a row fails, joined by "; ". The checks read the BMD, its bounds, the fit p-value, the scaled residual of the dose group nearest the BMD (the lower dose on a tie), and, under constant variance, the vehicle's scaled residual, the curve's SD over the vehicle replicates' sample SD and BMDS's test of one variance against a variance per dose group (test 2). The fit p-value is BMDS's test 4, the likelihood ratio test of the curve against a mean per dose group, (df + K) log(1 + (K - p) F / df) on a chi-square of K - p degrees of freedom, with F the curves' F-test of lack of fit ([Curves of named models](dose_response.md#dose-response-model-curves)): the fit p-value that BMDS, BMDExpress, NTP and ETAP read. Every parameter counts in p, as in the curves' AIC; BMDS's default leaves out those at a bound, such as a power curve's exponent at 1, which raises its fit p-value for such fits. A row that lacks what a check reads, such as the bounds of a cell type that supports only description, does not get that check.

| Set | Checks | What the guidance does with a failed check |
| --- | --- | --- |
| `"bmds_2026"` | Unusable: BMD or BMDL not computed (a BMDL of 0 counts). Questionable: fit p-value < 0.1; \|scaled residual near the BMD\| > 2; lowest dose / BMD or lowest dose / BMDL > 10; BMD / BMDL > 20; constant variance test p-value < 0.05. Warning: BMD or BMDL / highest dose > 1; lowest dose / BMD or BMDL > 3; BMD / BMDL > 3; curve's vehicle SD / observed vehicle SD > 1.5; \|scaled residual at the vehicle\| > 2 | BMDS recommends a model from those without questionable or unusable failures [@usepa2026b] |
| `"ntp_2018"` | BMD, BMDL or BMDU not computed, or beyond the search (a BMDL of 0 or a BMDU of infinity counts); BMD not below the highest dose; fit p-value <= 0.1; BMDU / BMDL >= 40; BMD more than 10-fold below the lowest dose (NTP keeps the gene) | A gene enters the gene-set analysis only without failures, except the last, which NTP reports and keeps [@ntp2018] |
| `"etap_2024"` | BMD or BMDL not computed; BMD not below the highest dose; fit p-value <= 0.1; BMD / BMDL >= 20 | A gene enters the transcriptomic point of departure only without failures [@usepa2024] |
| `"efsa_2022"` | BMD / BMDL > 20; BMDU / BMDL > 50; BMD more than 10 times below the lowest dose | Indicators for expert consideration, not exclusions [@efsa2022] |

In the hepatocytes of the example, 51 of the 88 genes that pass the gate pass ETAP's checks at 1.349 SD; 35 of the others fail its check of the fit p-value.

(rule-sets)=
## NTP's and ETAP's rule sets

A reviewer checks a transcriptomic point of departure against the workflow of an accepted document, and NTP 2018 and ETAP (US EPA 2024) both run BMDExpress 2 [@ntp2018; @usepa2024; @phillips2019], whose settings they leave partly to each platform or unstated. `rules="ntp_2018"` or `rules="etap_2024"` runs one document's workflow in every cell type that supports tests, as BMDExpress 2.30.520 runs it, on the counts of the cell type's samples:

```python
etap = cp.dose_response(
    cube, cp.Design(cube.samples, "~ 1"), dose=axis, rules="etap_2024", rng=1
)
settings = etap.rules("hepatocyte")
genes = etap.table("hepatocyte", "rule_set")
```

In the hepatocytes of the example, 59 of the 873 genes with a mean count of at least 5 pass the ANOVA pre-modelling test, so the cell type is analysed; 108 pass Williams' trend test and the fold change, and 59 of them pass ETAP's acceptance. The series has 3 animals per dose group where ETAP asks for 4, and its lowest step is half a log10 where ETAP asks for a full one, which the `outcome` of `rules("hepatocyte")` reports as deviations. The moderated tests and the gate run beside the rule set, unchanged.

| Setting | ETAP (US EPA 2024) | NTP 2018 | BMDExpress 2.30.520 |
| --- | --- | --- | --- |
| data set | one chemical, sex and tissue: the samples of one call in one cell type (Sec. 3.4.5.3) | one chemical: the samples of one call in one cell type (p. 5) | expressionDataConfigs: one file per data set |
| design minimums | 8 doses and a vehicle, spaced by half a log10 with the lowest step a full log10; 4 animals per dose group; a dose level with fewer than 2 passing samples is left out, and a cell type with fewer than 3 passing vehicle samples is not modelled (Sec. 3.4.1.3, 3.4.2 and 3.4.5.2) | none stated (pp. 10 to 11) |  |
| values | genes with a mean count of at least 5, counts per million over them, log2(CPM + 1) (Sec. 3.4.5.2) | as ETAP (p. 9: log2 values after normalization; Gwinn et al. 2020: log2(CPM + 1); ETAP's count filter) | logTransformation: BASE2 |
| ANOVA pre-modelling test | one-way ANOVA of the dose groups; the cell type is analysed when a gene's Benjamini-Hochberg adjusted p-value is below 0.05, and is inactive otherwise (Sec. 3.4.5.3) | one-way ANOVA of the dose groups; the cell type is analysed when a gene's Benjamini-Hochberg adjusted p-value is below 0.05 (p. 5) | anova prefilter: pValueCutoff 0.05, useMultipleTestingCorrection true |
| Williams' trend test | rising or falling, whichever statistic is larger, on 10,000 permutations; p < 0.05, not adjusted (Sec. 3.4.5.3; permutations not stated, Gwinn et al. 2020's) | as ETAP (pp. 5 to 6: thresholds set per platform; Gwinn et al. 2020, the S1500+ platform) | williams prefilter: numberOfPermutations 10000, pValueCutoff 0.05, useMultipleTestingCorrection false |
| fold change | at least 1.5-fold from the vehicle's mean at one dose or more, either way (Sec. 3.4.5.3) | as ETAP (pp. 5 to 6: set per platform; Gwinn et al. 2020: more than 1.5-fold, where ETAP's 1.5x or greater keeps 1.5 itself) | williams prefilter: useFoldChange true, foldChange 1.5 |
| models | linear, polynomial 2 (unrestricted), power, Hill and exponential 2 to 5, with the bounds of BMDS 2's programs (Sec. 3.4.5.3) | as ETAP (p. 6) | modelConfigs: poly 1, poly 2, power, hill, exp 2 to 5 |
| variance | constant (Sec. 3.4.5.3) | as ETAP (p. 7) | constantVariance true |
| power exponent | at least 1, at most 18 (Sec. 3.4.5.3) | at least 1, at most 18; NTP's equation allows exponents above 0 (p. 7; BMDExpress's default) | restrictPower true |
| benchmark response | 1.349 SD, the SD of the fitted curve as BMDExpress estimates it; the BMD at 1.349 times the vehicle replicates' SD, ETAP's wording, prints beside it (Sec. 3.4.5.3; Thomas et al. 2007) | 1 SD, the SD of the fitted curve under constant variance (p. 7) | ETAP: bmrFactor 1.349; NTP: bmrFactor 1 |
| bounds | one-sided 95% profile-likelihood bounds of the BMD, the ends of the piece of the likelihood's region that holds it (BMDS 2's programs in BMDExpress 2.30.520, observed) | as ETAP (BMDS 2's programs in BMDExpress 2.30.520, observed) | confidenceLevel 0.95 |
| model choice | the lowest AIC, counting the parameters not at a bound, among the models with a BMD, BMDL and BMDU, after leaving out a Hill fit whose k is below a third of the lowest dose (Sec. 3.4.5.3) | the lowest AIC, counting the parameters not at a bound, among the models with a BMD, BMDL and BMDU (p. 8) | ETAP: bestPolyTest 2; flagHillWithKParameter true, kParameterValue 3, bestModelSelectionWithFlaggedHill 2; NTP: bestPolyTest 2; flagHillWithKParameter false |
| fit test | BMDS's test 4: the likelihood ratio of the curve against a mean for each dose group (Sec. 3.4.5.3) | as ETAP (p. 9) | fitPValue |
| acceptance | BMD below the highest dose, fit p-value above 0.1 and BMD / BMDL below 20 (Sec. 3.4.5.3) | BMD, BMDL and BMDU computed, BMD below the highest dose, fit p-value above 0.1 and BMDU / BMDL below 40; a BMD more than 10-fold below the lowest dose is flagged and kept (pp. 8 to 9) | ETAP: categoryAnalysisConfigs: removeBMDGreaterHighDose true, bmdPValueCutoff 0.1, bmdBMDLRatioMin 20; NTP: categoryAnalysisConfigs: removeBMDGreaterHighDose true, bmdPValueCutoff 0.1, bmduBMDLRatioMin 40 |

`rules()` gives these settings with their sources, and for a cell type what each step found there. NTP 2018 leaves the prefilter's thresholds to each platform, and the NTP rule set takes those of NTP's study of the S1500+ platform [@gwinn2020], which match ETAP's: a p-value of [Williams' trend test](dose_response.md#williams-trend-test) below 0.05 on 10,000 permutations, and a 1.5-fold change. Where neither states a setting, it takes ETAP's, since ETAP states that its steps are consistent with NTP 2018. ETAP states no number of permutations; at 10,000, a p-value near 0.05 has a Monte Carlo SD of about 0.003.

- **The SD of the BMR.** ETAP sets the BMR to "1.349 * standard deviation of replicate vehicle control samples" [@usepa2024], and BMDExpress, which ETAP runs, takes the SD of the fitted curve: on a straight line, its BMD is 1.349 sqrt(RSS / n) / |b1| to every digit it prints. The rule set follows BMDExpress, so that its BMDs are those of ETAP's software, and `bmd_vehicle_sd` gives the BMD at ETAP's wording beside each, without bounds. Among the hepatocytes' accepted genes, the vehicle's reading gave a median of 0.96 times the BMD, with quartiles of 0.51 and 1.06.
- **The model choice.** BMDExpress's AIC counts the parameters that are not at a bound of BMDS 2's programs, so a power curve at n = 1 ties a straight line, and it takes the lowest AIC among the models whose BMD, BMDL and BMDU are numbers, computing the bounds of each model it passes over. ETAP leaves out a Hill fit whose k is below a third of the lowest dose. `model_choice` says why each gene's model was chosen.
- **The bounds.** BMDS 2 bounds a BMD by the piece of the profile's region that holds it, where CytePool's bounds otherwise reach the region's far ends ([Bounds by profile likelihood](#benchmark-dose-bounds)); under the documents' choices the bounds follow BMDS 2, so that NTP's BMDU/BMDL and ETAP's BMD/BMDL read what BMDExpress reads. On the rat livers of the validation, where both programs chose one model, the documents' acceptance of CytePool's own BMDs kept the genes that BMDExpress keeps for 211 of 213 under NTP and all 207 under ETAP ([Validation](../validation.md#validation-toxicology-rule-sets)).
- **The fit test.** BMDS's test 4, the likelihood ratio of the curve against a mean for each dose group on K minus the counted parameters, is BMDExpress's fit p-value; under BMDS's recommended model, test 4 counts the parameters that pybmds does not flag as bounded. On CytePool's own curves, the flag sets read test 4 with every mean parameter counted.
- **The curves.** The set `"bmdexpress"` holds the eight models of `"epa"` as BMDS 2's programs fit them in BMDExpress: the polynomial of degree 2 has no restriction on the signs of b1 and b2, so it can rise and then fall, and Hill's k and the rates of exponential 4 and 5 are bounded only from below.

The table `"rule_set"` has one row per gene the rule set kept, with the steps' statistics, the gene's model, its BMD and bounds at the document's BMR, the document's checks that the BMD fails and whether it is `accepted`; {api}`DoseResponseResult` lists the columns. The tables `"curves"` and `"benchmark_doses"` hold the rule set's curves of the genes that pass its prefilter. With `model=`, {api}`plot_dose_response` draws those curves over the means of the rule set's log2(CPM + 1) values, where the spline's figure draws the analysis's values. For curves fitted outside a cube with `models="bmdexpress"`, {api}`benchmark_doses` with `model="ntp_2018"` or `"etap_2024"` makes the document's choice, and {api}`BenchmarkDoses.accepted` gives its acceptance. A cell type where no gene passes the ANOVA pre-modelling test is inactive, as both documents state, and its table keeps the ANOVA columns alone. ETAP's minimums leave out a dose level of fewer than 2 passing samples and do not model a cell type of fewer than 3 passing vehicle samples. A cell type left with fewer than 3 dose groups is not modelled either, since the smallest model, a line, needs a spare group; one with a sample that has no counts in the genes the rule set keeps is not analysed, and its status names the sample, which ETAP's checks of sample quality would have removed. The summary's counts of each step are NaN where the step did not run.

The rule sets read each cell type's counts, as BMDExpress reads a platform's, so they leave out the analysis's TMM factors, sample weights and moderated variances, and they analyse one data set, such as the samples of one sex: a design of other terms is refused. Where many genes rise with dose, they take a larger share of each profile's counts, so the counts per million of the others fall along the doses, a trend that Williams' trend test reads and the gate's TMM factors remove ([Dose response per cell type](../guides/dose_response.ipynb)). Steps the documents name that CytePool does not apply are listed by `rules()` with the reason: the samples' sequencing quality and outliers, which belong to pooling; one gene per feature, since a cube's features are its genes; NTP's calibration of the prefilter on vehicle controls of each platform, which pseudobulk profiles lack; and models without a spare dose group, which CytePool does not fit [@ringblom2014].

(bmds-recommended-model)=
## BMDS's recommended model

`selection="bmds_2026"`, with `models="epa"` and a `bmr`, chooses each gene's model as BMDS recommends it, by the rule that pybmds 25.2 runs [@usepa2026b], outside both rule sets. Each of the six continuous models BMDS fits, linear, polynomial 2, power, Hill and exponential 3 and 5, is binned by the `"bmds_2026"` checks at the first BMR, with test 4 as the fit test; among the models with no questionable or unusable failure, BMDS takes the lowest AIC when their BMDLs lie within 3-fold of each other, and the lowest BMDL otherwise. A gene without such a model gets `"no model chosen"`, and `model_choice` gives the reason:

```python
bmds = cp.dose_response(
    cube,
    cp.Design(cube.samples, "~ 1"),
    dose=axis,
    models="epa",
    bmr=cp.BenchmarkResponse.sd(1),
    selection="bmds_2026",
    groups="hepatocyte",
)
recommended = bmds.table("hepatocyte", "benchmark_doses")
```

In the hepatocytes of the example, 63 of the 88 genes that pass the gate have no viable model at 1 SD; in 39 of them BMDS's test of one variance against a variance per dose group fails, a questionable failure of every model under the constant variance that BMDS's choice reads here.

(benchmark-doses-across-cell-types)=
## Benchmark doses across cell types

An SD-based BMR is a different change in each cell type: the SD of a pseudobulk profile shrinks as its cells add up, so 1.349 SD is a smaller change in a cell type of many cells per profile than in one of few, and its BMDs are lower for the same curve. In the example, 1.349 SD is a median change of 0.49 log2 units in the hepatocytes, with 300 cells per profile, and of 0.59 in the stellate cells, with 60. Compare cell types on an absolute BMR, and read `bmr_change` to see what an SD-based BMR amounts to in each. An absolute BMR compares like with like where counts suffice: log-CPM values compress the observed changes of genes with few counts, so that at 100 cells per profile a median of 0.40 of a true change was seen below 1 expected count per profile and 0.95 from 2 to 5 ([Validation](../validation.md#validation-simulate-dose-series)), more so in a cell type of few cells per profile. Compare the absolute BMDs of genes with enough counts in both cell types.

(potency-defaults)=
## Defaults and choices

- **The BMR.** None by default: `bmr` names it, as the guidance disagrees ([Benchmark doses](#benchmark-doses)); the 1 SD BMD is always added.
- **The SD of an SD-based BMR.** The curve's, as BMDS and BMDExpress estimate it, inside the bounds' profile.
- **The model.** Each gene's model of lowest AICc; `model=` names one model for every gene, `selection="bmds_2026"` takes BMDS's recommended model and a rule set its document's choice.
- **Bounds.** One-sided 95% profile-likelihood bounds; none for cell types that support only description, and {api}`benchmark_doses` refuses them for curves of dose groups without replicates.
- **The search.** From a hundredth of the lowest positive dose to 50 times the top dose.
- **Variance.** That of the curves: constant until simulations show that weighted fits keep their coverage ([Curves of named models](dose_response.md#dose-response-model-curves)). The curves are functions of dose, so the offset of a vehicle on the axis changes no BMD.

(potency-limits)=
## Limits and differences

- **The chosen model.** The bounds condition on the model chosen by AICc and on the direction of its change at the BMD, as BMDS's do, and so miss the uncertainty of the choice: on simulated series of 9 dose groups x 3, the BMDLs of a 1.1-fold change held the true BMD of 75% of the genes whose true shape lies in the `"epa"` set and whose true BMD lies within the doses, 86% where the chosen model contains the true shape and 72% where it does not ([Validation](../validation.md#validation-dose-response-calibration)). A curve averaged over models is not implemented yet.
- **Few replicates.** The chi-square threshold is approximate at a few replicates, even when the model is right: for Hill curves with normal errors and the Hill model named, at 9 dose groups x 3, the BMDLs of a change of 0.5 held the true BMD in 92.4% of datasets, the BMDUs in 90.4% and BMDL to BMDU in 82.8%, against 95%, 95% and 90%; at 8 replicates, 93.5%, 93.4% and 86.8% ([Validation](../validation.md#validation-dose-response-calibration)). On straight lines the SD-based BMDU covers less than the absolute one, as [the SD inside the profile](#benchmark-dose-bounds) says.
- **Bells.** The profile constrains the change at a dose, not the first crossing, so the BMDU of a bell can extend beyond its peak, where the curve reaches the BMR a second time.
- **Parameters at bounds.** The chi-square threshold is approximate where a parameter of the profiled curve sits at a bound, as Hill's exponent at 18 does for step-like responses.
- **The fit p-value.** The flag sets read test 4, as BMDS, BMDExpress, NTP and ETAP do; it rejects a correct curve more often than `lack_of_fit_p_value`, the F-test, at a few replicates: for a correct straight line at 9 doses x 3, 25% of the time where the F-test does 10% ([Curves of named models](dose_response.md#dose-response-model-curves)). Read `lack_of_fit_p_value` for a test at its nominal level. Where a parameter sits at a bound, BMDS's default count gives test 4 more degrees of freedom than CytePool's, so its check of fit can pass a curve that CytePool's fails.
- **BMDS's own rules.** `"bmds_2026"` follows the default rules that pybmds 25.2 runs, which differ from BMDS's user guide: the guide warns at BMD/BMDL > 5 rather than 3 and when the BMDU is not computed, and leaves out the vehicle's scaled residual [@usepa2026b].
- **References.** DRomics' SD has n - p in its denominator, BMDS's and CytePool's n; BMDS reports BMDs beyond 50 times the top dose that CytePool gives as not reached; pybmds's bound search can stop short of the threshold ([Validation](../validation.md#validation-benchmark-doses)).
- **The vehicle's SD.** ETAP's text defines the 1.349 SD as the vehicle replicates' SD, and `rules="etap_2024"` gives the BMD at that SD beside BMDExpress's ([NTP's and ETAP's rule sets](#rule-sets)), without bounds: under constant variance the vehicle's SD and the curve's estimate the same SD, so no likelihood holds the vehicle's fixed, and bounds that held it fixed covered the true BMD less often than stated ([Bounds by profile likelihood](#benchmark-dose-bounds)).
- **BMDS 2's fits.** BMDExpress runs BMDS 2's programs, which often stop short of a model's maximum, exponential 3 and 5 most of all, so on the rat livers of @gwinn2020 the rule sets' choice on CytePool's fits named BMDExpress's model for 73% to 75% of the probes, near FastBMD's 76% [@ewald2021], and on BMDExpress's own fits for every probe ([Validation](../validation.md#validation-toxicology-rule-sets)). Exponential 4 and 5 keep c at most 1,000, as in `"epa"`, where BMDS 2 leaves it free and can take it further on log2(CPM + 1) values whose vehicle level lies near 0.
- **Williams' p-values.** A p-value near 0.05 has a Monte Carlo SD of about 0.003 at 10,000 permutations, so a gene near the threshold can pass the prefilter at one seed and not at another, and BMDExpress fixes a seed of its own: on the rat livers, the rule sets' prefilter calls agreed with BMDExpress's for 99.7% of the probes.
- **Relative BMRs.** A share of a log2 level has no meaning for expression, and log2 CPM can be 0 or negative; a relative change of expression is an absolute BMR on log2 values.
- **Flags.** Each set is a convention of its guidance, without error control [@efsa2022].

(potency-checks)=
## Checks and further reading

Benchmark doses and their bounds are compared with DRomics, pybmds, scipy and mpmath, run as separate programs on the same inputs, and the flag sets with the same rules coded in R ([Validation](#validation-benchmark-doses)); their coverage of true BMDs is in [the calibration](#validation-dose-response-calibration). The rule sets are compared with BMDExpress 2.30.520 on the rat liver counts of @gwinn2020, and BMDS's recommended model with pybmds 25.2 ([Validation](#validation-toxicology-rule-sets)).

Further reading:

- @crump1984 defined the benchmark dose.
- @usepa2012 sets out benchmark dose modelling, with the 1 SD comparison and profile-likelihood bounds.
- @moerbeek2004 compared the coverage of profile-likelihood, delta-method and bootstrap bounds of BMDs.
- @thomas2007 argued for 1.349 SD as the BMR of genes whose adverse direction is unknown.
- @efsa2022 gives EFSA's guidance on benchmark dose modelling and its indicators; @fieller1954 derived the exact bound of a ratio.

[Gene-set tests](gene_set_tests.md) begins the part that turns the results for genes into results for pathways and regulators.
