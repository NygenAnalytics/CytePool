---
description: Where doses sit on the axis of a curve, what a number of dose groups supports, whether a dose series supports tests in each cell type, and dose series simulated from known curves.
---

# Dose response

[Composition tests](composition.md) asked whether a condition changes which cells a sample holds. A dose series asks of each cell type which genes respond to a dose and from which dose, and two things come before any test or curve: where each dose sits on the axis of a curve, which {api}`DoseAxis` records, and whether the samples of each cell type can support tests at all, which {api}`check_dose_design` reports. Series simulated from known curves with {api}`simulate_dose_series` show how well a dose method recovers the responses at a design before it is run. A dose response is fitted per cell type on the samples, as [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) fits a condition.

(dose-axes)=
## Dose axes and the vehicle

Where does the vehicle, the dose of 0, go on a dose axis? A curve of dose usually follows log10 of the dose, on which a half-log series takes equal steps, and 0 has no logarithm. {api}`DoseAxis` records the dose column of the sample table, its unit, what the dose measures and the axis, and places the vehicle in one of two ways:

| `vehicle` | Where the vehicle goes | What the design holds |
| --- | --- | --- |
| `"own_group"` (default) | In a group of its own, off the axis | A column `dose_treated`, 1 for a dose above 0; the spline of the treated doses, 0 for the vehicle |
| `"on_axis"` | At log10(offset) on a log10 axis, with `offset` added to every dose, or at 0 on a linear one | The spline of every sample's dose |

An offset decides where the vehicle sits and so the slope at the low end. DoseFinding's linear-in-log model adds one, `log(dose + off)` [@bornkamp2026], and sci-Plex-GxE added 0.001 to every dose, a value chosen from experiments whose data are not shown [@mcfaline2024]; {api}`DoseAxis` takes no default, and records the offset given. {api}`DoseAxis.positions` places doses on the axis, and {api}`DoseAxis.doses` reads positions back as doses, which prints knots in the dose's unit.

`metric` says what the dose measures, because doses of different metrics are not comparable without a conversion:

- `"administered"`: the amount given to the animal or person, as mg/kg;
- `"nominal"`: the amount added to the medium;
- `"measured"`: an amount measured in the medium, blood or tissue;
- `"internal"`: the amount at the target, estimated by a model.

{api}`DoseAxis.to_dict` records the axis, with its unit and metric, as plain values.

{api}`DoseAxis.spline_basis` places the knots on the doses of the samples, in the restricted cubic form by default, and {api}`DoseAxis.design_columns` gives the columns of each sample. Its default rule, `"levels"`, places the knots at quantiles of the distinct doses, each dose counted once, so a knot can fall between two doses. For a cube of hepatocytes whose sample table holds each sample's dose in `dose`, the vehicle and eight doses from 0.01 to 30 ug/kg:

```python
axis = cp.DoseAxis("dose", unit="ug/kg", metric="administered")
basis = axis.spline_basis(cube.samples, df=3)
print(axis.doses(basis.knots), axis.doses(basis.boundary_knots))
# [0.14422496 2.08008382] [1.e-02 3.e+01]
samples = cube.samples.with_columns(axis.design_columns(cube.samples, basis))
columns = ["dose_treated", "dose_s1", "dose_s2", "dose_s3"]
design = cp.Design(samples, "~ " + " + ".join(columns))
result = cp.limma_voom(
    cube,
    design,
    contrasts=columns,
    f_tests={
        "any_dose": columns,  # any dose against the vehicle
        "among_doses": columns[1:],  # any change among the treated doses
        "nonlinear": columns[2:],  # departure from a line in log10 dose
    },
)
```

The two interior knots sit at 0.14 and 2.1 ug/kg, between doses of the ladder, and the boundary knots at the lowest and highest doses. Each F-test has its table, such as `result.table("hepatocyte", "any_dose")`, whose `f_statistic`, `p_value` and `adjusted_p_value` columns test each gene: `any_dose` whether it responds at all, `among_doses` whether it changes along the treated doses, monotone or not, and `nonlinear` whether that change departs from a line in log10 dose. The tables of the spline columns, and the `log2_fold_change_dose_s1` to `log2_fold_change_dose_s3` columns of the F-tests' tables, hold coefficients that mean nothing one by one and change with the basis. `log2_fold_change_dose_treated` is the jump from the vehicle to the curve at the lowest dose, because every spline column is 0 at the lower boundary knot.

The fitted curve of each gene comes from the columns at a grid of doses and the coefficients in the table:

```python
import numpy as np

table = result.table("hepatocyte", "any_dose")
at = axis.design_columns({"dose": np.geomspace(0.01, 30.0, 50)}, basis)
curves = sum(np.outer(table[f"log2_fold_change_{name}"], at[name]) for name in columns)
```

Row i of `curves` is the fitted log2 fold change of gene i of the table against the vehicle, at each of the 50 doses.

How many spline columns? A spline of `df` columns needs at least `df + 1` treated doses, so `df` is at most the number of treated doses minus one; at that maximum the curve passes through the mean of every dose group, and `any_dose` is the F-test of a factor of dose groups against the vehicle. Fewer columns give a smoother curve, with fewer coefficients to estimate from 2 or 3 replicates per dose, as `df=3` on the eight doses above; for a time course of many ordered levels, the limma User's Guide recommends 3 to 5 (section 9.6.2). With `vehicle="on_axis"`, the design holds the spline columns alone, the vehicle counts among the doses, and `df` can be one more. The vehicle's rows of the spline columns change neither the three F-tests nor any fitted curve; they decide only what `dose_treated` and the intercept mean.

On a ladder of decades, such as 0.1, 1 and 10, the first restricted cubic column holds the whole numbers 0, 1 and 2, which {api}`Design.check` warns may be codes of groups (`numeric_codes`). The warning does not apply to the columns of a dose axis, and {api}`check_dose_design` takes the design of the other terms, so its report does not carry it.

(dose-groups)=
## What a number of dose groups supports

How much can a dose series say? The number of dose groups K, the vehicle included, sets it. A test of any dose against the vehicle has K - 1 degrees of freedom. A spline of dose with the vehicle in a group of its own runs along the K - 1 treated doses, so it has at most K - 2 columns, and K - 1 with the vehicle on the axis ([Dose axes and the vehicle](#dose-axes)). A curve of p parameters fitted to the means of K groups leaves K - p groups to test its fit, so a curve with a spare group needs p at most K - 1. Two documents set numbers of doses: EMA's qualification of MCP-Mod, a method for dose finding in phase 2 trials that tests candidate curve shapes and then fits one [@ema2014], and the standard methods of EPA's Transcriptomic Assessment Products (ETAP), for transcriptomic studies of chemicals in rats [@usepa2024].

| K | Tests | A curve with a spare group | What documents ask |
| --- | --- | --- | --- |
| 2 | The dose against the vehicle | None | |
| 3 | Any dose, on 2 degrees of freedom; each dose against the vehicle; a spline of 1 column | A line | MCP-Mod's test step, placebo included |
| 4 | A spline of up to 2 columns | An Emax curve, which rises to a maximum effect, of 3 parameters | MCP-Mod's modelling step |
| 5 | A spline of up to 3 columns | A Hill curve, of 4 parameters | |
| 9 | A spline of up to 7 columns | Any curve of up to 8 parameters | ETAP: 8 doses and a vehicle |

Replicates count too: n samples in each of K groups leave K(n - 1) residual degrees of freedom before moderation, 3 for a vehicle and two doses in duplicate, so the prior of the moderation ([Empirical Bayes moderation](empirical_bayes.md)) carries much of each gene's variance estimate.

(dose-design-check)=
## Checking a dose design

Which cell types of a dose series can be tested, and which can only be described? Filtering can leave a dose group without a sample in a depleted cell type [@hoffman2026a], a dose processed in a lane or plate of its own cannot be told apart from that batch, which breaks the control of false positives [@lun2017], and a dose given to a single animal has no replicate. {api}`check_dose_design` reports these per cell type before anything is fitted:

1. **Samples.** Each cell type uses the samples that its fit in {api}`limma_voom` would use, with the same `min_cells` and `min_counts`: at least 10 cells there by default, some counts, and at least `min_counts` counts summed over all features. The expression filter does not run.
2. **Dose groups.** The distinct doses of the design's samples, the vehicle first. A dose group that keeps no sample in a cell type is dropped there; one that keeps a single sample stays, and the doses table shows it. The replicates of a dose group are its samples or, when the design has random intercepts, the levels of each grouping among them: two samples of one animal at a dose, under `random="(1 | animal)"`, are one replicate.
3. **The dose-group model.** The design holds the terms other than dose, such as `"~ 1"` or `"~ sex"`, and the check adds an intercept and an indicator of each treated dose group that keeps a sample. The rank, by R's `qr()` rule, gives the residual degrees of freedom. A dose group is estimable when the design can compare it with the vehicle apart from its other terms: when its column lies outside the span of the other columns [@searle1971], which limma's `nonEstimable` reports with the column placed last. Random intercepts are checked on the samples used, as {api}`Design.check` checks them.
4. **The batch.** With `batch`, a column such as the lane or plate of each sample, a dose group is confounded with the batch when it is estimable and its difference from the vehicle is also a difference between batches. Batches of a single sample are pooled first, since such a batch cannot be told apart from its sample's own deviation, and a sample alone in its batch, when it is the only one, joins no batch, so a lane per sample confounds nothing. The rule finds exact confounding: a batch unbalanced across the doses without confounding them, as when samples are loaded into lanes in dose order, also biases the comparisons when the model leaves it out, so a known batch belongs in the design whenever it is unbalanced.

Each cell type gets one of three statuses:

| Status | When | What it supports |
| --- | --- | --- |
| `"inferential"` | An estimable treated dose group has two replicates or more, the model leaves residual degrees of freedom, and every dose group confounded with the batch is compared through a random intercept of it | Tests with p-values, and curves with bounds |
| `"descriptive"` | No estimable treated dose group has a second replicate; or no residual degrees of freedom are left; or a dose group is confounded with the batch, and the design does not have the batch as a random intercept over two or more batches of the vehicle and of each such group | Curves that describe the means, without p-values or bounds |
| `"refused"` | Every sample was left out; the design or the cell type has no vehicle sample; no treated sample is left; no treated dose group is estimable; or the random intercepts cannot be estimated beside the dose groups | No dose analysis |

The replicates that count are those of the treated doses: a replicated vehicle with a single sample at each dose leaves residual degrees of freedom, but no dose shows the variation between replicates given it. A batch that the design holds as a fixed term absorbs the dose groups it confounds, which then are not estimable. One that it holds as a random intercept compares them against the variance between batches when the vehicle and each such dose span two or more batches; with the vehicle alone in one lane, that variance rests on too few lanes, and the cell type stays descriptive.

For a cube of a series modelled on studies of TCDD (dioxin) in mouse liver, 24 mice given the vehicle or one of eight doses from 0.01 to 30 ug/kg, in lanes of two mice, with hepatocytes, stellate cells, which become scarce at the top dose, rare cholangiocytes and macrophages, which the vehicle mice lack:

```python
axis = cp.DoseAxis("dose", unit="ug/kg", metric="administered")
other_terms = cp.Design(cube.samples, "~ 1")
report = cp.check_dose_design(
    cube, other_terms, dose=axis, batch="lane", rules="etap_2024", sex="sex"
)
print(report)
flagged = report.doses.filter(group_id="stellate", n_samples_used=1)
missed = report.minimums.filter(met=False)
```

`batch`, `rules` and `sex` are optional: `cp.check_dose_design(cube, other_terms, dose=axis)` checks the samples, dose groups and replicates alone, as an in vitro or clinical series needs.

```text
DoseDesignReport: 4 cell types, 9 dose groups from 0 to 30 ug/kg, batch 'lane',
                  rules etap_2024
  hepatocyte: inferential; 9 of 9 dose groups keep a sample, 2 to 4 each; 15
              residual df; rules: deviation
  stellate: inferential; 9 of 9 dose groups keep a sample, 1 to 4 each; 14
            residual df; rules: deviation, ETAP excludes doses short of 2
            passing samples (M: 1 at 30 ug/kg)
  cholangiocyte: descriptive; 3 of 9 dose groups keep a sample, 1 each; 0
                 residual df; no treated dose group that can be compared with
                 the vehicle keeps a second sample, so no dose has a replicate;
                 rules: deviation, ETAP excludes doses short of 2 passing
                 samples (M: 1 at 0.01 ug/kg, 1 at 0.03 ug/kg, 0 at 0.1 ug/kg, 0
                 at 0.3 ug/kg, 0 at 1 ug/kg, 0 at 3 ug/kg, 0 at 10 ug/kg, 0 at
                 30 ug/kg)
  macrophage: refused; 8 of 9 dose groups keep a sample, 2 to 3 each; 12
              residual df; no vehicle sample is left (4 vehicle samples left
              out); compare the doses with each other with limma_voom; rules:
              deviation
```

The hepatocytes keep every mouse and 15 residual degrees of freedom. The stellate cells stay inferential though one of the two mice at 30 ug/kg has fewer than 10 of them: eight replicated groups estimate the variance, and `flagged` shows the group that keeps one mouse at 30 ug/kg. The cholangiocytes keep three mice, one per dose group, so their curves can only describe; the macrophages have no vehicle sample to compare with. The lanes, assigned at random, confound no dose.

Test the cell types that support tests, with the design of the spline columns from [Dose axes and the vehicle](#dose-axes):

```python
inferential = report.summary.filter(status="inferential")["group_id"].tolist()
result = cp.limma_voom(
    cube, design, groups=inferential, contrasts=columns, f_tests={"any_dose": columns}
)
```

The {api}`DoseDesignReport` holds four tables:

| Table | One row per | What to look at |
| --- | --- | --- |
| `summary` | Cell type | `status` and `reason`; `n_dose_groups`, the dose groups that keep a sample, and `n_dose_groups_estimable`, those the design can compare with the vehicle; `df_residual` |
| `doses` | Cell type and dose group | Groups with `n_samples_used` 1, or a low `n_cells_min` or `total_counts_min`; `estimable` False beside a kept group; `confounded_with_batch` |
| `samples` | Cell type and sample | `used`, and `reason` for a sample left out, as {api}`limma_voom` words it |
| `minimums` | Minimum of the rule set | Rows with `met` False, their `observed` text and `when_unmet`; `group_id` None marks a minimum of the design |

{api}`DoseDesignReport.status` reads the status of a cell type, and `report.design_check` holds {api}`Design.check` of the design given.

(dose-rule-minimums)=
## ETAP's design minimums

Which of EPA's design requirements does a dose series miss? The standard methods of EPA's Transcriptomic Assessment Products (ETAP) set minimums for the design and for the samples that pass quality control [@usepa2024]. With `rules="etap_2024"` and `sex`, the column of each sample's sex, `minimums` lists those that samples show, each met or not, with what ETAP does when it is unmet in `when_unmet`:

| `minimum` | Section | Required | When unmet, ETAP |
| --- | --- | --- | --- |
| `dose_levels` | 3.4.1.3 | At least 8 doses and a vehicle | departs from its design |
| `dose_spacing` | 3.4.1.3 | Half a log10 between doses, down from the highest, and the lowest dose a full log10 below the second lowest | departs from its design |
| `sexes` | 3.4.2 | Male and female animals | departs from its design |
| `animals_per_sex_and_dose` | 3.4.2 | At least 4 animals of each sex in each dose group, the vehicle's included | departs from its design |
| `passing_samples_per_dose` | 3.4.5.2 | At least 2 passing samples at each dose level | excludes the dose level and goes on |
| `passing_vehicle_samples` | 3.4.5.2 | At least 3 passing vehicle samples | does not model the dose response |

A step counts as half a log10 when its log10 lies within 0.05 of 0.5, and as a full log10 within 0.05 of 1: one decimal place of log10, the precision of "half-log10", which holds both steps of a 1-3-10 series, 0.48 and 0.52. Animals are counted among the design's samples, one per animal, and passing samples are those the cell type uses. ETAP defines a dataset by tissue, sex and exposure regimen, so the minimums of passing samples apply to each cell type and sex. {api}`DoseDesignReport.rules_status` is `"deviation"` when a minimum of the design or the cell type's vehicle minimum is unmet, and `"met"` otherwise: a dose level that ETAP excludes is listed, in `minimums` and in the printout, without changing it, since ETAP's procedure goes on without the level. The TCDD series above misses three minimums of the design: one sex, 2 or 3 mice per dose, and steps of 0.48 log10 at its low end, where ETAP asks for a full log10. The other four rows of `missed` belong to cell types: ETAP excludes the stellate cells' 30 ug/kg, where one mouse passes, and every dose level of the cholangiocytes, and the cholangiocytes and macrophages keep 1 and 0 of the 3 passing vehicle samples ETAP needs to model a dose response.

(simulated-dose-series)=
## Simulated dose series with known curves

How well does a dose method find the genes that respond, the shape of each response and the dose at which it starts, at a given design? Only data whose truth is known can say, and the one single-cell dose benchmark compared tests on cells, without curves [@nault2022]. {api}`simulate_dose_series` draws a cube of pseudobulk counts from known curves of each gene's expression along dose, and {api}`DoseTruth` holds the truth. Here the spline F-test of [Dose axes and the vehicle](#dose-axes) runs on a simulated series, and the truth says which genes it should call:

```python
import numpy as np

cube, truth = cp.simulate_dose_series([0, 0.01, 0.1, 1, 10, 30], replicates=3, rng=1)
axis = cp.DoseAxis("dose", unit=truth.unit, metric=truth.metric)
basis = axis.spline_basis(cube.samples, df=3)
samples = cube.samples.with_columns(axis.design_columns(cube.samples, basis))
columns = ["dose_treated", "dose_s1", "dose_s2", "dose_s3"]
design = cp.Design(samples, "~ " + " + ".join(columns))
result = cp.limma_voom(cube, design, contrasts=columns, f_tests={"any_dose": columns})

tested = result.table("cell_type_1", "any_dose")
called = tested.filter(tested["adjusted_p_value"] < 0.05)["feature_id"]
genes = truth.table("cell_type_1")
genes = genes.filter(np.isin(genes["feature_id"], tested["feature_id"]))
hit = np.isin(genes["feature_id"], called)
size = np.abs(genes["max_log2_change"])
responsive, flat = size > 0, size == 0
print(
    f"{hit[responsive].sum()} of {responsive.sum()} responsive genes called, "
    f"{hit[flat].sum()} of {flat.sum()} flat"
)
for change in (0.5, 1, 2):
    chosen = size == change
    print(f"maximal change {change}: {hit[chosen].sum()} of {chosen.sum()} called")
# 59 of 113 responsive genes called, 2 of 461 flat
# maximal change 0.5: 4 of 32 called
# maximal change 1: 21 of 38 called
# maximal change 2: 34 of 43 called
```

Of the 574 genes that pass the expression filter of {api}`limma_voom`, the F-test of any dose calls 59 of the 113 that respond and 2 of the 461 flat ones at 5% FDR: with 3 replicates per dose and 100 cells per profile, about half of the responses go unseen while false calls stay rare. Half of the misses are changes of 0.5 log2 units, of which 4 in 32 are called, against 34 in 43 changes of 2. The same lines with `replicates=4` or `cells=30` show what another design gains or loses before it is run.

The cube has one group per cell type and one sample per replicate, in dose order. Its sample table holds `sample_id`, such as `d2_r1` for the first replicate of the third dose; `dose`, with 0 for the vehicle, which {api}`DoseAxis` reads as it stands; `dose_group`, `d0` to `d5` here, which sorts in dose order with the vehicle first, for designs with a factor of dose groups; and `donor`. The layer `counts` and `n_cells` hold each profile's counts and cells, and the provenance records every argument. Each profile sums the counts of a random number of cells:

| Level | Model | Default spread |
| --- | --- | --- |
| A gene's expression at the vehicle | Expected counts per cell, log-normal, partly shared by the cell types | Median 0.1 counts per cell; SD 1.5 (natural log) shared and 0.5 per cell type |
| A gene's variability | A log-normal factor on `biological_sd` and `donor_sd` | SD 0.3 (natural log) |
| Cells per profile | Poisson, with a log-normal mean of `cells` times the `composition` factor of the dose | The Poisson mean's median 4.4% below `cells`, its coefficient of variation 0.31; counting adds 1 / `cells` to the squared coefficient of variation |
| Sequencing depth | A log-normal factor with mean 1 | SD 0.2 (natural log) |
| Biological effect | Normal in log2 units, per replicate (`biological_sd`), plus a donor effect shared by the doses when donors are crossed with dose (`donor_sd`) | `biological_sd` 0.25 |
| Counts | Negative binomial, the sum of the profile's cells | Dispersion 0.5 per cell, so 0.5 / n for n cells |

Profiles of fewer cells vary more beyond their counts, so precision follows the cells per profile along dose, as it does for a cell type that a dose depletes. Replicates differ by effects on every gene, as in the simulations of muscat and scDesign3 [@crowell2020; @song2024]. A share `responsive` of the genes respond, with these curves of the change in log2 expression from the vehicle, where M is the maximal change (`max_log2_change`), E the EC50 (`ec50`), the dose of half the maximal change, n the slope (`slope`), P the peak dose (`peak_dose`), w the bell's SD along log10 dose (`peak_width`) and T the top dose:

| `shape` | Change at dose d | Fitted families that contain it | Default share |
| --- | --- | --- | --- |
| `hill` | M / (1 + (E/d)^n) | BMDS's Hill, drc's LL.4, DRomics' Hill | 30% |
| `exponential` | M (1 - 2^(-d/E)) | BMDS's exponential 4, DRomics' exponential | 15% |
| `linear` | M d / T | BMDS's linear and power, DRomics' linear | 5% |
| `power` | M (d/T)^n | BMDS's power | 10% |
| `log_gauss_probit` (a symmetric bell) | M exp(-(log10(d/P) / w)^2 / 2) | DRomics' log-Gauss-probit | 20% |
| `frechet` | M 2^(-(E/d)^n) | none | 10% |
| `step` | M from d = E on, 0 below | none | 10% |

The fitted families are the curves that BMDS [@usepa2026], drc's four-parameter log-logistic model LL.4 [@ritz2015] and DRomics [@delignettemuller2023] fit; Hill, sigmoid Emax and four-parameter log-logistic curves are one family [@kappenberg2023]. The Fréchet curve, drc's Weibull type 1 with a negative slope parameter, is the Weibull curve of BMDS's exponential 5 model turned end for end on a log-dose axis: flat at low doses, then a steep rise and a slow approach to its plateau. Steps and Fréchet curves lie outside every family, so a method is not checked only on curves of its own form. A truth inside a family can still lie outside a tool's bounds: on doses scaled to the top dose, BMDS bounds the Hill EC50 at 5, which 9% to 11% of Hill truths exceed, and the rate of its exponential 4 model below 100, an EC50 above 0.007 times the top dose.

The maximal changes `max_log2_changes`, 0.5, 1 and 2 log2 units by default, a 1.4-, 2- and 4-fold change, come in equal shares, half rises and half falls.

{api}`DoseTruth.table` gives each gene's shape, parameters, expected counts per cell at the vehicle, `expression_sd` and two benchmark doses, the doses at which the true curve first departs from the vehicle by a stated change, the benchmark response (BMR) [@crump1984]: `bmd_expression_sd` at one `expression_sd` and `bmd_fold_change` at a fold change of 1.1 in either direction, BMDS's two defaults for continuous data, one SD of the control and a change of 10% [@usepa2026], with the fold change taken on the log2 scale. {api}`DoseTruth.curve` gives the true change at any dose, and {api}`DoseTruth.benchmark_doses` the benchmark dose at any change in log2 expression, such as the 1.349 SDs of EPA's Transcriptomic Assessment Products (ETAP); a rise of 10% is `log2_change=np.log2(1.1)`, not 0.1. A method's estimates can then be compared with the truth gene by gene: their bias, and the coverage of their intervals, the share of intervals that contain the true value over simulated datasets.

The benchmark dose of a 1.1-fold change of Hill, exponential, Fréchet and step curves is log-uniform from an eighth of the log10 span of the positive doses below the lowest dose to an eighth above the top dose, so that a tenth lies below the doses and a tenth above. The EC50 follows from it, so about a quarter of Hill and exponential EC50s lie above the top dose, where the doses never reach half the maximal change, and over a third at 4 dose groups. Slopes are log-uniform from 0.5 to 4, and from 1 to 4 for power curves, which reach their maximal change at the top dose, as linear curves do.

Bells peak within the doses, with an SD along log10 dose of 0.75 to 1.5 times the gap between the doses on either side of the peak, so that 98% to 99% reach half their peak at two or more doses, at six designs of 3 to 9 dose groups, evenly spaced or not. Their benchmark dose, the lower crossing, lies below the lowest dose more often in sparse designs: in 31% of bells at nine dose groups (the vehicle and eight half-log doses) and 94% at four (the vehicle and three doses a decade apart) ([Validation](../validation.md#validation-simulate-dose-series)).

`expression_sd` is the SD of true expression between vehicle replicates, from biological and donor variation, before counting, and every truth is a change in true expression. Counting widens the SD of observed log-CPM values most for genes with few counts: at 100 cells per profile, 3.6 times for genes with 1 to 5 expected counts at the vehicle and 1.2 times above 100. The prior count of log-CPM values compresses the observed changes of the same genes, falls more than rises: at the top dose and 100 cells, to 0.40 of the true change below 1 expected count, 0.80 from 1 to 2 and 0.95 from 2 to 5, and to 0.99 or more from 5 counts on ([Validation](../validation.md#validation-simulate-dose-series)).

For SDs on the observed scale, simulate the same seed with many vehicle replicates, such as `replicates=[2000, 3, 3, 3, 3, 3]`, which keeps the truth, and pass a multiple of each gene's observed SD to {api}`DoseTruth.benchmark_doses`. That pairs an observed SD with the true curve, so compare such benchmark doses only for genes with at least 5 expected counts at the vehicle and the top dose, or draw the observed curve, the mean log-CPM at each dose, from a series with many replicates of every dose.

The random draws come in independent streams, so a cell type's curves and expression stay the same when `replicates`, `cells`, `composition`, `biological_sd` or `donor_sd` change, or another cell type is added: designs compare on the same genes. Doses multiplied by a factor give EC50s, peak doses and benchmark doses multiplied by it, and the same counts.

(dose-response-defaults)=
## Defaults and choices

- **The vehicle.** In a group of its own. On the axis, at 0 on a linear axis, or on a log10 axis only at an offset that is given and recorded ([Dose axes and the vehicle](#dose-axes)).
- **The samples of a cell type.** Those of its fit in {api}`limma_voom`: `min_cells=10` and `min_counts=0`, with the same meaning ([Checking a dose design](#dose-design-check)).
- **Dose groups of one sample.** Kept and shown in the doses table; a group is dropped only when it keeps no sample. ETAP excludes levels with fewer than 2 passing samples, which `minimums` lists.
- **Replication.** Judged at the treated doses: a cell type whose estimable treated dose groups each keep one sample, or the samples of one level of a random intercept's grouping, is descriptive, whatever the vehicle keeps.
- **Batches of one sample.** Pooled before the batch is checked; a sample alone in its batch, when it is the only one, joins no batch.
- **A batch as a random intercept.** It compares the dose groups it confounds with the vehicle when the vehicle and each such group span two or more batches.
- **Cells per profile.** Reported, with no threshold beyond `min_cells`. In muscat's benchmark of differential state tests, methods on pseudobulk profiles performed decently from about 100 cells per cell type and sample, with moderate gains beyond [@crowell2020], which makes `n_cells_min` worth reading.

(dose-response-limits)=
## Limits and differences

- **Samples are the replicates, unless a random intercept names the unit.** With random intercepts, the replicates of a dose group are the levels of each grouping among its samples. Under `"~ 1"` or a fixed term of the unit, such as `"~ patient"`, repeated samples of one participant at a dose count as replicates: pool them first with {api}`Cube.collapse_samples`, by participant and dose, or name the participant as a random intercept. A design that names the animal as a random intercept, with one animal per dose group, the vehicle's included, is refused with advice to pool.
- **Several compounds.** The dose groups are the distinct doses of all the design's samples, so two compounds tested at the same concentrations would count each other's wells as replicates. Check one compound at a time, with a design of that compound's samples and the vehicle's: with `s = cube.samples`, `cp.Design(s.filter((s["compound"] == "A") | (s["dose"] == 0)), "~ 1")`.
- **Batches unbalanced across doses.** The batch rule finds exact confounding only; add a known batch to the design whenever it is unbalanced across the doses ([Checking a dose design](#dose-design-check)).
- **Doses measured per sample.** Measured exposures, one per sample, make a dose group of every sample, so most groups hold one sample and the check calls the cell type descriptive.
- **ETAP's other requirements.** Its sequencing quality metrics, species, strain, route, duration and age are not checked: samples do not show them.
- **The expression filter.** It does not run, so a cell type that the check calls inferential can still stop in its fit when no gene passes the filter.
- **Fits of descriptive designs.** {api}`limma_voom` fits a design that the check calls descriptive whenever residual degrees of freedom are left: in GSE208161, one participant per dose level sampled before and after dosing [@sonigra2022], 13 of 18 clusters keep one residual degree of freedom, the placebo participant's change from baseline.
- **Doses that differ by rounding.** {api}`DoseAxis.spline_basis` and {api}`check_dose_design` refuse two doses that differ by less than 1e-8 of the dose, as `0.1 * 3` and `0.3` do, which a conversion of units can leave as two doses where there is one: round the dose column first.
- **What a simulated series leaves out.** In {api}`simulate_dose_series`, a gene's biological SD is the same at every dose, the constant variance that BMDS fits by default; genes are independent given their curves, so gene-set methods meet no correlation between genes; each cell type draws its responsive genes on its own, so few genes respond in several cell types; bells are symmetric, and no curve is hormetic; the counts hold no ambient RNA; and library sizes move with dose, by a median of 0.08 log2 at the top dose of default series, which TMM removes for the most part, leaving about 0.01 log2 on the flat genes at the top dose.

(dose-response-checks)=
## Checks and further reading

Dose axes, dose designs and simulated dose series are compared with R, limma, scanpy, mpmath and published data, run as separate programs on the same inputs:

- {api}`DoseAxis`: positions, knots, columns, designs, F-tests and fitted curves against R 4.4.3's `ns()`, Hmisc's `rcspline.eval`, `model.matrix`, `lm` and `predict`, exact (1e-10); moderated F-tests of `limma_trend` against limma 3.62.1's, exact on `ns()`'s columns and close (1e-6) on Hmisc's, with every column counted, and equal in the two forms ([Validation](#validation-dose-axes)).
- {api}`check_dose_design`: cells, counts and the samples used per dose group against scanpy 1.12.4 and pandas; the residual degrees of freedom, estimable dose groups and groups confounded with a batch against R 4.4.3's `qr()` and limma 3.62.1's `nonEstimable`; statuses and ETAP's minimums against the same rules coded in R; and GSE208161, where every cluster is descriptive, all identical ([Validation](#validation-dose-design)).
- {api}`simulate_dose_series` and {api}`DoseTruth`: true curves, EC50s and benchmark doses against mpmath 1.3.0, exact (1e-10); the means and variances of cells and counts within 4 standard errors of their exact values; the same series at 1, 2 and 8 threads ([Validation](#validation-simulate-dose-series)).

Further reading:

- @lun2017 showed that plate effects confounded with the condition break the control of false positives, and that summing the counts within each plate restores it.
- @crowell2020 benchmarked differential state methods on simulated and real multi-sample data, and found pseudobulk methods performing decently from about 100 cells per cell type and sample.
- @usepa2024 sets out EPA's standard methods for transcriptomic assessment products, with the design minimums of its Sections 3.4.1.3, 3.4.2 and 3.4.5.2.
- @ema2014 qualified MCP-Mod for phase 2 dose finding, with its minimum numbers of doses.
- @nault2022 benchmarked tests of single-cell dose series simulated from the EPA's benchmark-dose equations; @crowell2020 and @song2024 describe the simulators of muscat and scDesign3; @crump1984 introduced the benchmark dose, and @kappenberg2023 reviews the design and analysis of 5,670 toxicological dose-response analyses.

[Gene-set tests](gene_set_tests.md) begins the part that turns the results for genes into results for pathways and regulators.
