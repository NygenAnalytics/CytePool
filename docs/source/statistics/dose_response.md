---
description: Dose axes, what a dose series supports, which genes and signatures respond to dose per cell type and how, a gate checked against noise, from which dose, curves of named models and simulated series.
---

# Dose response

[Composition tests](composition.md) asked whether a condition changes which cells a sample holds. A dose series asks of each cell type which genes respond to a dose and how, and two things come before any test or curve: where each dose sits on the axis of a curve, which {api}`DoseAxis` records, and whether the samples of each cell type can support tests at all, which {api}`check_dose_design` reports. {api}`dose_response` then tests every cell type that supports tests, with a fitted curve and its band for each gene, curves of named models such as BMDS's, tests of trends, of each dose against the vehicle and of candidate shapes of response, a gate that decides which genes count as responding, which {api}`permutation_check` checks against permuted doses, and the lowest dose from which each gene responds, and series simulated from known curves with {api}`simulate_dose_series` show how well a dose method recovers the responses at a design before it is run. [Dose response per cell type](../guides/dose_response.ipynb) runs the steps on a simulated series.

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
        "spline": columns,  # any dose against the vehicle
        "among_doses": columns[1:],  # any change among the treated doses
        "nonlinearity": columns[2:],  # departure from a line in log10 dose
    },
)
```

The two interior knots sit at 0.14 and 2.1 ug/kg, between doses of the ladder, and the boundary knots at the lowest and highest doses. Each F-test has its table, such as `result.table("hepatocyte", "spline")`, whose `f_statistic`, `p_value` and `adjusted_p_value` columns test each gene: `spline` whether it responds at all, `among_doses` whether it changes along the treated doses, monotone or not, and `nonlinearity` whether that change departs from a line in log10 dose. The tables of the spline columns, and the `log2_fold_change_dose_s1` to `log2_fold_change_dose_s3` columns of the F-tests' tables, hold coefficients that mean nothing one by one and change with the basis. `log2_fold_change_dose_treated` is the jump from the vehicle to the curve at the lowest dose, because every spline column is 0 at the lower boundary knot.

The fitted curve of each gene comes from the columns at a grid of doses and the coefficients in the table:

```python
import numpy as np

table = result.table("hepatocyte", "spline")
at = axis.design_columns({"dose": np.geomspace(0.01, 30.0, 50)}, basis)
curves = sum(np.outer(table[f"log2_fold_change_{name}"], at[name]) for name in columns)
```

Row i of `curves` is the fitted log2 fold change of gene i of the table against the vehicle, at each of the 50 doses.

How many spline columns? A spline of `df` columns needs at least `df + 1` treated doses, so `df` is at most the number of treated doses minus one; at that maximum the curve passes through the mean of every dose group, and `spline` is the F-test of a factor of dose groups against the vehicle. Fewer columns give a smoother curve, with fewer coefficients to estimate from 2 or 3 replicates per dose, as `df=3` on the eight doses above; for a time course of many ordered levels, the limma User's Guide recommends 3 to 5 (section 9.6.2). With `vehicle="on_axis"`, the design holds the spline columns alone, the vehicle counts among the doses, and `df` can be one more. The vehicle's rows of the spline columns change neither the three F-tests nor any fitted curve; they decide only what `dose_treated` and the intercept mean.

{api}`dose_response` builds these tests itself, in every cell type at once, as contrasts of a model of the dose groups whose variance the four tests share, beside the other terms of the design ([Which genes respond to dose](#dose-response-tests)); the columns here serve models it does not build, such as a spline that differs by another term, dose by sex, or a spline on knots of your own.

On a ladder of decades, such as 0.1, 1 and 10, the first restricted cubic column holds the whole numbers 0, 1 and 2, which {api}`Design.check` warns may be codes of groups (`numeric_codes`). The warning does not apply to the columns of a dose axis, and {api}`check_dose_design` takes the design of the other terms, so its report does not carry it.

(dose-groups)=
## What a number of dose groups supports

How much can a dose series say? The number of dose groups K, the vehicle included, sets it. A test of any dose against the vehicle has K - 1 degrees of freedom. A spline of dose with the vehicle in a group of its own runs along the K - 1 treated doses, so it has at most K - 2 columns, and K - 1 with the vehicle on the axis ([Dose axes and the vehicle](#dose-axes)). A curve of p parameters fitted to the means of K groups leaves K - p groups to test its fit, so a curve with a spare group needs p at most K - 1. Two documents set numbers of doses: EMA's qualification of MCP-Mod (multiple comparison procedure and modelling), a method for dose finding in phase 2 trials that tests candidate curve shapes and then fits one [@ema2014], and the standard methods of EPA's Transcriptomic Assessment Products (ETAP), for transcriptomic studies of chemicals in rats [@usepa2024].

| K | Tests | A curve with a spare group | What documents ask |
| --- | --- | --- | --- |
| 2 | The dose against the vehicle | None | |
| 3 | Any dose, on 2 degrees of freedom; each dose against the vehicle; a spline of 1 column | A line; exponential 2 | MCP-Mod's test step, placebo included |
| 4 | A spline of up to 2 columns | An Emax curve, which rises to a maximum effect, of 3 parameters; the quadratic, power, exponential 3 and 4 and DRomics' exponential | MCP-Mod's modelling step |
| 5 | A spline of up to 3 columns | A Hill curve, of 4 parameters; exponential 5, the log-logistic curve and the 4-parameter Gauss-probit curves | |
| 6 | A spline of up to 4 columns | The 5-parameter Gauss-probit curves, which rise and then fall, and a check of whether one dose group drives a turn | |
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

{api}`dose_response` runs this check itself, with the same `batch`, `min_cells` and `min_counts`, and tests the cell types it calls inferential ([Which genes respond to dose](#dose-response-tests)).

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

(dose-response-tests)=
## Which genes respond to dose

Which genes respond to the dose in each cell type, and how? {api}`dose_response` answers for every cell type of a cube in one call. It runs {api}`check_dose_design` and keeps its report as `result.design_report`, so the check needs no call of its own; fits one moderated model of the dose groups in each cell type that supports tests, by {api}`limma_voom` (the default), {api}`limma_trend` or {api}`dream_voom`; and runs four tests as contrasts of that one fit, so that a gene's tests share its one moderated variance:

| Test | Question | What is tested | Numerator df |
| --- | --- | --- | --- |
| `spline` | Does the gene respond at all, along a smooth curve of dose? | A natural cubic spline of dose fitted to the dose groups' means, with `dose_treated` when the vehicle is a group of its own | The spline's columns, plus 1 with the vehicle in a group of its own |
| `nonlinearity` | Does the response bend away from a straight line? | The spline's restricted cubic columns after the first | The spline's columns minus 1 |
| `linear_trend` | Does expression rise or fall along the axis, and how fast? | The slope of a straight line fitted to the dose groups' means | 1, a moderated t-test |
| `dose_groups` | Does any dose differ from the vehicle? | Every treated dose against the vehicle | The treated doses compared |

Read `spline` first, the default gate ([Which genes count as responding](#dose-response-gate)): it asks whether a gene responds at all. The linear trend and the non-linearity test describe the responses it finds, and `dose_groups` asks the same question as `spline` without a curve. The dose-group test is the one-way analysis of variance of the dose groups with moderated variances, the moderated counterpart of the ANOVA that the genomic dose-response approach of the National Toxicology Program runs as a gate: at least one gene must pass it at a Benjamini-Hochberg FDR below 0.05 for a data set to be analysed further [@ntp2018].

The spline is fitted by least squares to the means of the dose groups that a cell type compares, each group weighted by its samples, so that each of its coefficients is a contrast of the dose groups' differences from the vehicle. Without observation weights or other terms in the design, these are the coefficients of the spline fitted to the samples themselves, as the limma User's Guide fits a spline of time (section 9.6.2); under voom's weights the means are those of the weighted fit [@law2014]. The knots are placed once, on the doses of the design's samples, by {api}`DoseAxis.spline_basis` with its rule `"levels"`, so that the knots and curves of the cell types compare. By default the spline has 2 columns, 3 knots, when the axis holds 4 doses or more, and 1 otherwise: on simulated series of the vehicle and 8 doses, 2 columns found as many responsive genes as 3, and more than 1 column or a spline through every dose group ([Validation](../validation.md#validation-dose-response-calibration)). `spline_df` sets another size. A spline with as many coefficients as the dose groups a cell type compares passes through every group's mean, and its F-test is the dose-group F-test.

The linear trend's slope, the `log2_fold_change` of its table, is the change in log2 expression per log10 unit of dose, or per unit of dose on a linear axis. With the vehicle in a group of its own, the line runs through the treated doses only: a gene that jumps at the lowest dose and then stays flat has a slope near 0, and its jump from the vehicle is what `spline` and `dose_groups` test. With the vehicle on the axis, the line runs through every dose, the vehicle at its offset, as the regression on log dose of Tukey's trend test does [@tukey1985], and the slope is per log10 unit of the dose plus the offset.

For a series simulated from known curves ([Simulated dose series with known curves](#simulated-dose-series)), the vehicle and eight doses from 0.01 to 30 ug/kg in three replicates, with hepatocytes, stellate cells that fall to a tenth at the top dose and rare cholangiocytes:

```python
cube, truth = cp.simulate_dose_series(
    [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    replicates=3,
    cell_types=["hepatocyte", "stellate", "cholangiocyte"],
    cells=[300, 60, 6],
    composition={"stellate": [1, 1, 1, 1, 1, 1, 1, 1, 0.1]},
    rng=7,
)
axis = cp.DoseAxis("dose", unit="ug/kg", metric="administered")
result = cp.dose_response(cube, cp.Design(cube.samples, "~ 1"), dose=axis)
print(result)
spline = result.table("hepatocyte", "spline")
responding = spline.filter(spline["adjusted_p_value"] <= 0.05)
```

```text
DoseResponseResult (limma_voom): 3 cell types, 2 tested; tests: spline,
    nonlinearity, linear_trend, dose_groups
  dose: ug/kg (administered) on a log10 axis, the vehicle in a group of its own;
        spline of 2 columns, knots at 0.548 ug/kg, boundary knots at 0.01 and 30
        ug/kg; linear_trend slopes per log10 unit of dose
  gate: spline test, Benjamini-Hochberg within each cell type, at 5% FDR
  hepatocyte: 9 dose groups, 27 samples, 817 genes; 88 pass the gate; calls at
              5% FDR: spline 88, nonlinearity 31, linear_trend 81, dose_groups
              77
  stellate: 8 dose groups, 24 samples, 494 genes; 43 pass the gate; calls at 5%
            FDR: spline 43, nonlinearity 9, linear_trend 38, dose_groups 34
  cholangiocyte: not tested, refused: no vehicle sample is left (3 vehicle
                 samples left out); compare the doses with each other with
                 limma_voom
```

The {api}`DoseResponseResult` prints the axis, the spline's knots in doses, the unit of the linear trend's slope and the gate, which decides which genes count as responding ([Which genes count as responding](#dose-response-gate)), and for each cell type the dose groups it compares, its samples, the genes it tests, the genes that pass the gate and the genes each test calls at 5% FDR. The hepatocytes compare all 9 dose groups and the spline calls 88 genes, 85 of which respond in truth. The stellate cells keep no sample at 30 ug/kg, so they compare 8 dose groups, and their dose-group test has one degree of freedom fewer. The cholangiocytes keep no vehicle sample of 10 cells or more, so the check refuses them, and the summary gives its reason.

The result's tables, which {api}`DoseResponseResult.table` gives by cell type and name, and its summary:

| Table | One row per | What to look at |
| --- | --- | --- |
| `table(g, "spline")`, `"nonlinearity"`, `"dose_groups"` | Gene | `f_statistic`, `p_value`, `adjusted_p_value` and `adjusted_p_value_across_groups`; `df`, the denominator degrees of freedom |
| `table(g, "linear_trend")` | Gene | `log2_fold_change`, the slope; `t`, `p_value` and the adjusted p-values |
| `table(g, "each_dose")` | Gene and treated dose | Each dose against the vehicle: `log2_fold_change`, `standard_error`, `t`, `p_value` and `adjusted_p_value`, adjusted within the dose |
| `table(g, "gate")` | Gene | `passes_gate`, the gate's call; `p_value` and `adjusted_p_value` of the gate's test |
| `result.summary` | Cell type | `status`, the check's; `tested` and `reason`; `n_dose_groups_tested`; `tests_not_run`; `n_genes_passing_gate` |

{api}`DoseResponseResult.fit` computes a cell type's fit again, with the values, weights and moderated statistics behind its tables. `adjusted_p_value` adjusts the p-values of a cell type and test by Benjamini-Hochberg, as limma's `topTable` does, and `adjusted_p_value_across_groups` adjusts each test over every cell type that ran it, one family per test, for a report of calls across cell types. The tests of each dose are adjusted within the cell type and dose only. For each dose against the vehicle with control over a gene's several doses, as Dunnett's test gives it, pass the cell type's moderated fit and its dose contrasts to {api}`multiple_contrast_test` ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts) shows how); to call the doses across genes, use {api}`DoseResponseResult.lowest_effective_doses` ([From which dose a gene responds](#lowest-effective-dose)).

A cell type compares the dose groups that keep a sample there and that the design can compare with the vehicle; the provenance lists them with their samples. A test whose contrasts a cell type's doses cannot estimate is not run there, such as a spline of 2 columns on the vehicle and two treated doses, and `tests_not_run` names it, with the reason in the record and in the error of `result.table`. When the dose groups are few, two tests can ask the same question: with the vehicle and two treated doses, a spline of one column and `dose_treated` pass through the three groups' means, so the spline's F-test is the dose-group test; with the vehicle on the axis and three doses, a spline of one column is the straight line. By default a cell type then runs the gate's test, then the first of `spline`, `dose_groups` and `linear_trend`, and lists the others as not run; a test named in `tests` runs whatever it repeats.

(dose-response-curves)=
## Curves with a band

What shape does a response take? {api}`DoseResponseResult.curve` gives the fitted spline of chosen genes at any dose within those a cell type compares, as a change in log2 expression from the vehicle's mean, with a pointwise 95% band, and {api}`plot_dose_response` draws it beside each dose's estimate:

```python
from cytepool.plotting import plot_dose_response

genes = responding.sort("p_value")["feature_id"][:4].tolist()
curve = result.curve("hepatocyte", genes)
ax = plot_dose_response(result, "hepatocyte", genes=genes)
```

`curve` has one row per gene and dose, with `log2_fold_change`, `standard_error`, `ci_lower` and `ci_upper`, by default at 100 doses evenly spaced along the axis over the doses the cell type compares; `at` takes doses of your own, and genes may be named by symbol. At dose x, the curve is a combination a(x)'theta of the dose groups' differences from the vehicle, theta; its standard error is (a(x)' V a(x))^(1/2) times the gene's posterior standard deviation, with V the unscaled covariance of theta, each gene's own under voom's weights, and the band is the estimate plus and minus the 0.975 quantile of t on the degrees of freedom of the gene's moderated t-tests, as limma's `topTable(confint = TRUE)` forms intervals. The band is pointwise and covers the spline: at each dose, 95% of such bands contain the spline that least squares fits to the true means of the dose groups, which is the true curve where the response has the spline's shape. Where the spline cannot follow a response, as at a step or the turn of a bell, the band can miss the true curve, so read it beside each dose's estimate. On series of the vehicle and 8 half-log doses with 3 replicates, the bands held the true change at the doses for 95% of the doses of flat genes but 87% of those of responsive genes, 80% for bells and 81% for steps, while each dose's own 95% interval held it for 95% ([Validation](../validation.md#validation-dose-response-calibration)). Over the whole range at once, the curve stays inside the band less often. With the vehicle in a group of its own the curve is 0 at the vehicle, exactly; with the vehicle on the axis the spline need not pass through the vehicle's mean, and the curve's value at dose 0 shows how far it misses. `curve` refuses doses outside those the cell type compares, so the stellate cells above have no curve beyond 10 ug/kg, and it gives none for results of {api}`dream_voom`, whose fits do not keep the degrees of freedom of each point of a curve.

(dose-response-contrasts)=
## Trends, doses against the vehicle and candidate shapes

Does a gene's response grow with dose, which doses differ from the vehicle, and does the response follow one of a few expected shapes? The spline test says that a gene responds, not how. Three multiple contrast tests of the same fit answer these questions, each from the largest of several correlated moderated t-statistics of the gene, whose p-value comes from the multivariate t distribution ([Several contrasts of one gene](multiple_testing.md#correlated-tests)). They cost far more than the F-tests, so they run only when named in `tests`: on 20,000 genes of the vehicle and 7 doses in three replicates, on two threads, Dunnett's test and the MCP step each added about 10 s to the 0.9 s of the default tests and the Williams-type test 23 s; with 3 doses the MCP step took 0.6 ms per gene and the other two under 0.1 ms ([Timings](../validation.md#validation-timings)).

| Test | Question | Contrasts of the dose groups | Read it when |
| --- | --- | --- | --- |
| `dunnett` | Does any dose differ from the vehicle, and which differs most? | Each dose against the vehicle [@dunnett1955] | The doses themselves matter; each dose's own p-value comes from {api}`multiple_contrast_test` (below) |
| `williams` | Does the response grow with dose? | The vehicle against the mean of the top one, two or more doses, each dose weighted by its samples [@bretz2006] | Responses are expected to rise or fall in dose order |
| `mcp` | Does the response follow one of a few expected shapes? | The contrast that best detects each candidate shape: the MCP step, the multiple comparison procedure of MCP-Mod [@bretz2005; @pinheiro2014] | A few shapes are expected, from earlier studies or the pharmacology |

Each test is two-sided, so it finds rises and falls. A gene's `p_value` is adjusted for its own contrasts, the single-step max-t test: the chance that the largest of the gene's t-statistics is at least as extreme as its own [@hothorn2008]. `adjusted_p_value` adjusts these p-values over the cell type's genes by Benjamini-Hochberg, and `adjusted_p_value_across_groups` over every cell type that ran the test. Each contrast's standard error and the contrasts' correlation come from the gene's own covariance of its doses' estimates, voom's weights included, so the three tests refuse limma's approximate standard errors (`contrast_standard_errors="limma"`), which give no such covariance. Under {api}`dream_voom` each dose has its own degrees of freedom, and the multivariate t takes the smallest of the gene's ([Limits and differences](#dose-response-limits)). The Williams-type test is not Williams' trend test of the rule sets of the National Toxicology Program and EPA's Transcriptomic Assessment Products, which compares each dose with the vehicle after fitting means that change in dose order ([Dunnett's and Williams-type contrasts](multiple_testing.md#dunnett-and-williams-contrasts)).

{api}`CandidateShapes` holds the guesses of the MCP step, each a curve of the change from the vehicle along dose in one of DoseFinding's model families [@bornkamp2026]: Emax, d / (ED50 + d); sigmoid Emax, the Hill equation d^h / (ED50^h + d^h); exponential, exp(d / delta) - 1, BMDS's exponential 2 model, a rise that grows with dose; and beta, an umbrella shape that rises to a peak and falls. The candidate exponential shape rises late, as the simulator's `power` curves do; the simulator's `exponential` curves, BMDS's exponential 4, level off as Emax shapes do ([Simulated dose series with known curves](#simulated-dose-series)). DoseFinding's advice for MCP-Mod is that 3 to 7 shapes from 2 to 4 families usually suffice, with no linear shape, which an Emax shape of large ED50 approximates, and that designs have three treated doses or more, with 4 to 7 recommended [@bornkamp2026]. EMA's qualification asks for the vehicle and two doses for the test step, which `"mcp"` runs, and three for the modelling step, not implemented yet [@ema2014]. A shape becomes the contrast c of the dose groups that maximises the non-centrality of its t-statistic, c'mu / sqrt(c'Sc), for the shape's means mu at the doses and the covariance S of the groups' means: for groups of n_k samples, c_k is proportional to n_k (mu_k - the size-weighted mean of mu), which {api}`optimal_contrasts` computes, as DoseFinding's `optContr` does. A shape scaled or shifted gives the same contrast.

By default, `dose_response` spreads five shapes over the design's doses with {api}`CandidateShapes.spread`: between the lowest and highest treated dose, along the axis's positions, two Emax shapes with ED50s a third and two thirds of the way, a sigmoid Emax shape of Hill slope 3 halfway, an exponential shape that reaches half its change at the top dose D at 0.75 D, and a beta shape that peaks halfway, is half its peak halfway between the lowest dose and the peak, and has DoseFinding's scale of 1.2 D. The beta shape and the slope come from the share of a true response's squared non-centrality, which sets its power, that the best of the shapes' contrasts keeps: on simulated bells at 9 dose groups, a median 0.60 with the beta shape, against 0.36 with DoseFinding's quadratic umbrella or with none, and Hill slopes of 2 to 4 moved the medians by under 0.01. {api}`CandidateShapes.means` gives each shape's means at any doses, scaled as DoseFinding's `getResp` scales them, {api}`CandidateShapes.names` names them, and {api}`CandidateShapes.to_dict` records them. On the series of [Which genes respond to dose](#dose-response-tests):

```python
result = cp.dose_response(
    cube,
    cp.Design(cube.samples, "~ 1"),
    dose=axis,
    tests=("spline", "dunnett", "williams", "mcp"),
    rng=1,
)
print(result)
mcp = result.table("hepatocyte", "mcp")
```

```text
DoseResponseResult (limma_voom): 3 cell types, 2 tested; tests: spline, dunnett,
    williams, mcp
  dose: ug/kg (administered) on a log10 axis, the vehicle in a group of its own;
        spline of 2 columns, knots at 0.548 ug/kg, boundary knots at 0.01 and 30
        ug/kg
  gate: spline test, Benjamini-Hochberg within each cell type, at 5% FDR
  hepatocyte: 9 dose groups, 27 samples, 817 genes; 88 pass the gate; calls at
              5% FDR: spline 88, dunnett 69, williams 73, mcp 92
  stellate: 8 dose groups, 24 samples, 494 genes; 43 pass the gate; calls at 5%
            FDR: spline 43, dunnett 24, williams 26, mcp 49
  cholangiocyte: not tested, refused: no vehicle sample is left (3 vehicle
                 samples left out); compare the doses with each other with
                 limma_voom
```

In the hepatocytes, the MCP step calls 92 genes, 89 of which respond in truth, the spline 88 (85), the Williams-type test 73 (69) and Dunnett's test 69 (67). Of the 33 responsive genes with bell-shaped responses, the spline calls 21, the MCP step 19, Dunnett's test 17 and the Williams-type test 12: pooling the top doses misses responses that turn back down. Over simulated series of the vehicle and 8 doses, the MCP step found 42% and 49% of the responsive genes with 2 and 3 replicates, the spline 39% and 47%, the Williams-type test 27% and 35% and Dunnett's test 23% and 33% ([Validation](../validation.md#validation-dose-response-calibration)). The table of each test has one row per gene: `contrast`, the contrast of the gene's largest \|t\|, such as `"10 - vehicle"`, `"3 to 30 - vehicle"` or a shape's name, `"emax_1"`; its `t`; `df`, the degrees of freedom of the multivariate t; `p_value` and `p_value_error`, its Monte Carlo standard error, at most 1% of it; and the adjusted p-values. For the MCP step, `contrast` names the candidate whose contrast best matches the observed changes, not a fitted shape: the shapes' contrasts are strongly correlated, and fitting and choosing a shape is MCP-Mod's modelling step, not implemented yet. The weights of every contrast on the treated doses are in the cell type's entry of `result.provenance["groups"]`, under `contrast_tests`, and {api}`DoseResponseResult.candidates` gives the shapes. `rng` sets the Monte Carlo error of the p-values: `dose_response` records one seed in `result.settings["seed"]`, and each cell type and test draws from a generator seeded by it, so that `rng=` the recorded seed gives the same p-values whatever the other cell types, the other tests named and the threads; tests of two contrasts or fewer use no random numbers.

The tables give each gene's p-value over all its doses or shapes. Each dose's own p-value, adjusted for the gene's doses as Dunnett's test adjusts it, comes from {api}`multiple_contrast_test` on the cell type's fit, whose estimates include each dose against the vehicle:

```python
import numpy as np

from cytepool.stats.multiple_contrasts import multiple_contrast_test

fit = result.fit("hepatocyte")
names = list(fit.moderated.fit.coefficient_names)
# The fit's estimates: each dose against the vehicle, then the spline's contrasts.
doses = {n: np.eye(len(names))[j] for j, n in enumerate(names) if n.startswith("dose_")}
per_dose = multiple_contrast_test(fit.moderated, contrasts=doses, rng=1)
at_ten = per_dose.table(contrast="dose_10", feature_ids=fit.features["feature_id"])
```

`at_ten` has each gene's `estimate`, `standard_error`, `t`, `p_value` and `p_value_error` at 10 ug/kg, and the smallest of a gene's p-values over its doses is its p-value in the `"dunnett"` table, within their Monte Carlo errors. To call the doses across genes, use {api}`DoseResponseResult.lowest_effective_doses` ([From which dose a gene responds](#lowest-effective-dose)).

To test shapes of your own, give their parameters in the dose's unit, such as ED50s from an earlier study:

```python
# The beta shape peaks at 36 x 0.5 / (0.5 + 2) = 7.2 ug/kg.
shapes = cp.CandidateShapes(
    emax=[0.3, 3.0], sigmoid_emax=[(1.0, 2.0)], beta=[(0.5, 2.0)], beta_scale=36.0
)
guessed = cp.dose_response(
    cube,
    cp.Design(cube.samples, "~ 1"),
    dose=axis,
    tests="mcp",
    candidates=shapes,
    rng=1,
)
```

With the vehicle and K - 1 treated doses, the contrasts of the dose groups point in at most K - 1 directions, so more shapes than treated doses cannot all differ: five shapes on two treated doses span two dimensions, and the printout says so. The MCP step still tests every shape, at the cost of the multiplicity of shapes that duplicate one another; at three dose groups, EMA's minimum for the MCP step [@ema2014], Dunnett's test asks nearly the same. A shape that does not change at a cell type's doses, such as a narrow peak at a dose that the cell type does not keep, is left out there, and the record says why. MCP-Mod applied gene by gene to expression data at eight concentrations found that the sigmoid Emax model fitted the less noisy genes best and simpler models the noisy ones [@duda2022]; here the shapes give only contrasts, and fitting them, MCP-Mod's modelling step, is not implemented yet.

(williams-trend-test)=
## Williams' trend test as BMDExpress runs it

Which genes would NTP's and ETAP's workflows fit curves to? Both keep a gene whose mean rises or falls steadily with dose, by the two-sided Williams trend test of BMDExpress 2 with a fold change [@ntp2018; @usepa2024; @phillips2019]. {api}`williams_trend_test` computes that test for every gene of a samples x genes matrix of log-scale values, such as log2(CPM + 1), and returns a {api}`WilliamsTrendTest`:

```python
from cytepool.stats.trend_tests import williams_trend_test

test = williams_trend_test(doses, log2_values, rng=1)
test.table(feature_ids=genes)  # feature_id, direction, statistic, p_value
```

{api}`WilliamsTrendTest.table` gives one row per gene: the `direction` the data take, `up` or `down`, the `statistic` and its `p_value`; the result also holds each gene's `isotonic_means`, the fitted means of the dose groups in that direction.

Isotonic regression fits the dose groups' means, the vehicle included, once rising and once falling: each fit is the closest sequence of means in that order, weighted by the groups' sizes, found by pooling adjacent groups that break the order [@bartholomew1959]. A side's statistic is its fitted top dose minus its fitted vehicle, over the SD pooled within the dose groups times sqrt(1/n_top + 1/n_0), and the gene's `direction` is the side with the larger statistic. Williams [-@williams1971] fitted the treated groups alone and compared them with the vehicle's observed mean; BMDExpress 2.30.520 fits the vehicle with them, as enumerating every relabelling of small designs showed: of 18 forms of the test, only this one gave BMDExpress's p-values.

The p-value comes from random relabellings of the samples' doses, shared by all genes. Of the relabellings whose larger statistic points the gene's way, it is the share that reach the gene's statistic, counting the observed labelling among them: (b + 1) / (n + 1) [@phipson2010]. About half the relabellings point each way, so 10,000 permutations give a smallest p-value near 2 / 10,000, and a p-value near 0.05 a Monte Carlo SD of about 0.003. The result records its seed, and passing the seed back as `rng` gives the same p-values; BMDExpress fixes a seed of its own that its options do not expose, so the two programs agree within Monte Carlo error rather than to the digit ([Validation](#validation-toxicology-rule-sets)).

The Williams-type test of `dose_response(tests=["williams"])` asks a related question another way: it contrasts the vehicle with the pooled top doses in the moderated linear model of each cell type, and refers the largest contrast to the multivariate t distribution, so it uses moderated variances and voom's weights and needs no permutations ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts)). Williams' trend test as BMDExpress runs it uses unmoderated values, and both documents apply it to every gene without adjusting for their number: with BMDExpress's Williams test and a 1.5-fold change, vehicle-only zebrafish embryos assigned to 4 mock concentrations of 6 samples each gave a transcriptomic point of departure in 93% of runs [@lee2024]. The ANOVA pre-modelling test that the rule sets run first adjusts across genes, so that a data set without responsive genes passes it at most 5% of the time.

(dose-response-gate)=
## Which genes count as responding

Which genes count as responding to the dose, and what does that call rest on? A dose series runs several tests of each gene, and the genes that respond go on to curves, gene-set roll-ups and points of departure. One test decides: the gate, a test of any response at a stated false discovery rate, run after the expression filter. The filter reads each gene's counts and library sizes, never the doses, the kind of filter that leaves the error rate of the test after it in place [@bourgon2010]. {api}`dose_response` takes the gate as `gate`, records it and prints it:

| `gate` | What it asks of a gene |
| --- | --- |
| `{"test": "spline"}`, the default | Does it respond along a smooth curve of dose? |
| `{"test": "dose_groups"}` | Does any dose group differ from the vehicle? The default when fewer than two doses are on the axis, where no spline is placed |
| `{"test": "dunnett"}`, `"williams"` or `"mcp"` | Does any dose differ from the vehicle, does the response grow in dose order, or does it follow a candidate shape? |
| `{"fold_change": 1.5}` | Does it change by more than 1.5-fold, up or down, at a dose? |

`"fdr"` sets the level, 0.05 by default: a gene passes when the Benjamini-Hochberg adjusted p-value of the gate's test, over the cell type's genes, is at most the level. The gate's test runs whether `tests` names it or not. `"linear_trend"` and `"nonlinearity"` cannot gate, since they test a straight line or a departure from one, not any response.

```python
gate = result.table("hepatocyte", "gate")
large = cp.dose_response(
    cube, cp.Design(cube.samples, "~ 1"), dose=axis, gate={"fold_change": 1.5}
)
passed = large.table("hepatocyte", "gate")["passes_gate"]
print(f"default gate: {gate['passes_gate'].sum()} genes; 1.5-fold gate: {passed.sum()}")
```

```text
default gate: 88 genes; 1.5-fold gate: 24
```

The table `"gate"` has one row per gene tested: `p_value`, the p-value of the gate's test, `adjusted_p_value` and `passes_gate`. `result.summary` counts the genes that pass in `n_genes_passing_gate`, NaN where the gate's test did not run, which says nothing of whether the cell type responds. {api}`DoseResponseResult.gate` gives the gate as `dose_response(..., gate=result.gate)` takes it back, and the provenance records its adjustment for multiple testing (`gate_adjustment`) beside its test, level and fold change.

A fold change enters each dose's test against the vehicle as its null hypothesis, by TREAT [@mccarthy2009], instead of a cut on the estimated changes after the test, as the rules of NTP and EPA's Transcriptomic Assessment Products apply it: a cut controls no error rate for the size of the change. Each dose's TREAT p-value tests whether its change exceeds the fold change; a gene's p-value is the smallest of its doses' times the doses, a Bonferroni adjustment that holds whatever the correlation between a gene's doses, which the gate records as the test `"each_dose"`; Benjamini-Hochberg then runs across genes. In the hepatocytes above, 93 of the genes tested change by more than 1.5-fold at one of the doses, in truth. The 1.5-fold gate passes 24 of them and no other gene; the default gate with a cut at 1.5-fold on the estimated changes keeps 86 genes, 77 of those 93, 7 whose true change never exceeds 1.5-fold and 2 flat genes. TREAT asks for evidence that a change exceeds the threshold, so a change must lie well above it to pass, and thresholds of 1.2 to 1.5 are usual ([Testing against a fold-change threshold](empirical_bayes.md#treat-test)). A gate with a fold change needs {api}`limma_voom` or {api}`limma_trend` ([Limits and differences](#dose-response-limits)).

The gate takes the place of BMDExpress's prefilter, the step that picks the genes that go on to curves. BMDExpress's default prefilter takes an unadjusted Williams p-value and cuts the estimated fold change; the gate adjusts its p-values and puts any fold change inside the test. It is not the ANOVA gate of NTP's and EPA's rules, a test of the data set that at least one gene must pass before the set is analysed further [@ntp2018]. {api}`DoseResponseResult.lowest_effective_doses` calls the doses of the genes the gate passes ([From which dose a gene responds](#lowest-effective-dose)); curves keep every gene that the expression filter keeps, and gene-set roll-ups and points of departure, which will admit only the genes the gate passes, are not implemented yet.

(dose-permutation-check)=
## Could the calls come from noise?

Could the genes that a gate passes come from noise at this design? In zebrafish embryos exposed to the vehicle alone and placed at random in 4 mock concentration groups of 6 samples, BMDExpress's default prefilter, a Williams trend test with a 1.5-fold cut, gave a transcriptomic point of departure, the benchmark dose of the 25th most sensitive gene, in 93% of runs [@lee2024]. With the concentration labels of a real series permuted 100 times, a workflow whose prefilter took an unadjusted Williams p-value found responsive genes in every permutation, where one with a false-discovery-rate-adjusted Williams test found none in nearly all [@bruns2026]. {api}`permutation_check` asks the question of a result, at its own design:

```python
check = cp.permutation_check(result, n_permutations=100, rng=1)
print(check)
```

```text
PermutationCheck of the gate (spline test, Benjamini-Hochberg within each cell
    type, at 5% FDR): 3 cell types, 2 checked; seed 1
  hepatocyte: 88 genes pass the gate; over 100 random runs of 1.1e21
              assignments, the gate passed genes in 3 runs (0.03), 0.04 genes
              per run; p = 0.0099, the smallest these runs can give
  stellate: 43 genes pass the gate; over 100 random runs of 3.7e17 assignments,
            the gate passed genes in 3 runs (0.03), 0.09 genes per run; p =
            0.0099, the smallest these runs can give
  cholangiocyte: not checked: cell type 'cholangiocyte' was not tested
                 (refused): no vehicle sample is left (3 vehicle samples left
                 out); compare the doses with each other with limma_voom
```

For each cell type, the dose labels of the samples it uses are permuted among the samples that the design treats alike, {api}`dose_response` runs again on that cell type with each permuted labelling, the result's settings and its gate, and the genes that pass the gate are counted. The check returns a {api}`PermutationCheck`: {api}`PermutationCheck.summary` gives for each cell type the genes that the true doses pass, the share of runs that pass genes, the mean genes per run, and the permutation p-value (1 + b) / (1 + B) for the b of B runs that pass at least as many genes as the true doses [@phipson2010]. {api}`PermutationCheck.table` gives each run's count, and the provenance each run's assignment, so that any run can be made again with the result's seed.

Read the p-value first. Under the null hypothesis that no gene responds to dose, every assignment of the labels is as likely as the true one, so a small p-value says that the true doses pass more genes than permuted ones: not all of the gate's calls are noise, though the p-value does not say which calls are false. Above, the gate passed genes in 3 of 100 runs in each cell type, and no run passed as many genes as the true doses, so p is 1/101, the smallest that 100 runs can give. The share of runs that pass genes says how often the gate passes genes with these samples' doses shuffled. It estimates how often noise alone passes genes only where no gene responds, as in vehicle-only data or a cell type whose p-value is large, and then only for these samples: on 100 simulated series of the vehicle and three doses in duplicate in which no gene responds, the share averaged 0.16 and varied between series with a standard deviation of 0.08 ([Validation](../validation.md#validation-dose-response-calibration)). How often a gate passes noise at a design takes many series, as the Validation page runs them, or a vehicle-only series.

Which samples count as alike follows the design. Without random effects the units are the samples, in blocks with the same row of the design's matrix apart from dose, so that `"~ sex"` permutes the doses within each sex, and a numeric covariate leaves each sample in a block of its own, which the check lists as not checked. With a random intercept such as `(1 | animal)`, animals that each received one dose are permuted whole, in blocks of animals with the same rows, so that each run keeps the samples and animals of every dose group; donors whose cells received every dose keep their samples, and the doses are permuted within each donor. Each run keeps the doses of the design, so the spline's knots, the sizes of the dose groups and the genes that the expression filter keeps stay those of the result. Before the runs, the true doses run once through the same path and must give the result's gate, the same genes with p-values equal up to a relative 1e-9 and the same calls, so that the runs repeat the analysis that the result records; another CPU's arithmetic moves the p-values by about 1e-13.

A design with few samples allows few assignments: the vehicle and two doses in duplicate allow 6! / (2! 2! 2!) = 90. When the assignments other than the true one number at most `n_permutations`, the check runs each of them once, which makes its p-value exact. Some runs pass the true doses' genes whatever the data, because the gate cannot tell their labels from the true ones:

- under the dose-group test, and the spline when it passes through every group's mean, as at the vehicle and two doses, the runs that only rename dose groups of equal size;
- under the gates that compare each dose with the vehicle, with a fold change or by Dunnett's test, the runs that rename the treated doses;
- at one treated dose, the swap of the dose and the vehicle, under every gate, since each tests their difference on both sides;
- under the spline, the reflection of doses that sit symmetrically on the axis, such as 0, 5 and 10 on a linear axis with the vehicle on it, when the reflected groups are of equal size.

The check counts these ties among its runs, as the runs whose gate gives the true doses' p-values up to rounding, and the summary gives the smallest p-value the runs can give as `smallest_possible_p_value`, (1 + t) / (1 + B) for the t of B runs that tie. The Williams-type test and the MCP step tell the doses apart. Dunnett's test of three or more treated doses computes its p-values by Monte Carlo integration, which leaves its renamings apart by that error, so they do not count. For the dose-group gate:

| Design | Assignments | Smallest possible p-value of the dose-group gate, every assignment run |
| --- | --- | --- |
| The vehicle and 1 dose x 2 | 6 | 2/6, 0.33 |
| The vehicle and 1 dose x 3 | 20 | 2/20, 0.10 |
| The vehicle and 1 dose x 4 | 70 | 2/70, 0.029 |
| The vehicle and 2 doses x 2 | 90 | 6/90, 0.067 |
| The vehicle and 2 doses x 3 | 1,680 | 6/1,680, 0.0036 |
| The vehicle and 3 doses x 2 | 2,520 | 24/2,520, 0.0095 |

At the vehicle and two doses in duplicate, with the vehicle in a group of its own as the default axis places it, no p-value of the default gate falls below 0.05, however strongly the genes respond. With random runs, B is `n_permutations`: the smallest possible p-value is 0.0099 at the default 100 runs when no run ties, and the last two designs reach theirs only with `n_permutations` of 1,679 and 2,519, which run every assignment.

On series in which no gene responds, simulated at 4 and 9 dose groups with 2 and 3 replicates, the check's runs passed genes as often as the series themselves did, the excess at 2 replicates included (0.162 of runs against 0.154 of the other series at 4 dose groups in duplicate), and its p-value was at most 0.05 in 1.5% to 6.0% of cell types ([Validation](../validation.md#validation-dose-response-calibration)).

A series of vehicle-only samples that a design places in mock dose groups, as @lee2024 studied, is a null series by construction: {api}`dose_response` and this check run it as they run any series, and every gene the gate passes is a false call. On 200 simulated series of their design, 4 mock dose groups of 6 samples, the default gate passed a gene in 17 of 400 cell types, 4.3%, and a prefilter like the one they studied, a Williams-type p-value below 0.05 with no adjustment across genes and a 1.5-fold change at a dose, passed a gene in every one ([Validation](../validation.md#validation-dose-response-calibration)).

Each run fits the gate's test alone, and takes the time of the cell type's analysis and of that test. On 20,000 genes and 24 samples, on two threads, a run took 0.70 s with the spline gate, so 100 runs take about 70 s per cell type; 1.4 s with a fold change, which fits the cell type once more for TREAT; and 24 s with the Williams-type gate, whose multivariate t probabilities take about a millisecond per gene. Under {api}`dream_voom`, each run takes the time of dream's fit ([Timings](../validation.md#validation-timings)).

(lowest-effective-dose)=
## From which dose a gene responds

From which dose does each gene respond? ICH E4 asks that the lowest dose recommended for use itself show an effect [@ich1994], and toxicology reads the lowest observed effect level (LOEL) of each endpoint. Calling each dose against the vehicle in every gene would multiply the false calls by the doses and the genes, so {api}`DoseResponseResult.lowest_effective_doses` calls them in two stages, as stageR does [@vandenberge2017]:

1. The gate, the spline by default, passes R of the G genes it tested at its false discovery rate, alpha ([Which genes count as responding](#dose-response-gate)).
2. In each gene it passes, the doses' moderated t-tests against the vehicle, those of the table `each_dose`, are called by Holm's step-down procedure at the level R alpha / G that the gate leaves, with Shaffer's first step: the first multiplier is the gene's doses minus one, not the doses, since the gate has rejected that every dose equals the vehicle [@holm1979; @shaffer1986]. Under a gate with a fold change, each dose's TREAT p-value against that fold change takes the place of its t-test.

The lowest effective dose is the lowest dose called. The calls control the overall false discovery rate (OFDR) at alpha: the expected share of the genes passed with at least one false call, the gate's or a dose's, proved under independence between genes [@heller2009] and expected wherever Benjamini-Hochberg controls the false discovery rate [@vandenberge2017]. They do not control the share of false calls among all gene-dose calls, which counts a gene once for each dose called. [Calling contrasts across genes](multiple_testing.md#stage-wise-testing) gives the formula.

```python
lowest = result.lowest_effective_doses("hepatocyte")
called = lowest.filter(lowest["passes_gate"])
top = called.sort("stage_wise_adjusted_p_value").take(range(5))
for gene, dose, change, p, n in zip(
    top["feature_id"],
    top["lowest_effective_dose"],
    top["log2_fold_change"],
    top["stage_wise_adjusted_p_value"],
    top["n_doses_called"],
):
    print(f"{gene}  {dose:>4g} ug/kg  {change:+.1f} log2  p {p:.2g}  {n} of 8 doses")
```

```text
gene_0631    30 ug/kg  +1.9 log2  p 6.3e-10  1 of 8 doses
gene_0172    30 ug/kg  +2.4 log2  p 1.6e-08  1 of 8 doses
gene_0556    10 ug/kg  +2.1 log2  p 3.6e-08  2 of 8 doses
gene_0412     3 ug/kg  -2.0 log2  p 5.5e-07  3 of 8 doses
gene_0040    10 ug/kg  +1.7 log2  p 7.1e-07  2 of 8 doses
```

The table has one row per gene tested: `passes_gate` is the gate's call; `lowest_effective_dose` gives the lowest dose called, in the axis's unit, with that dose's `log2_fold_change` and `stage_wise_adjusted_p_value`; and `n_doses_called` counts the doses called. The lines above show the five genes of smallest stage-wise adjusted p-value. Of the 88 genes the gate passes in the hepatocytes, 69 have a dose called. A dose is called when its stage-wise adjusted p-value is at most the gate's level, as stageR counts, and the values hold only at that level, since R depends on it: another level, or another test of any response, is another gate, such as `gate={"test": "williams", "fdr": 0.1}` in {api}`dose_response`, rather than the values read against another threshold.

An effective dose here is a dose whose change is detectable at the design, with no threshold on its size but the gate's fold change: the clinical minimum effective dose usually adds a clinically relevant change, as MCP-Mod's target dose does [@bornkamp2026]. It is a tested dose, not a potency between doses, and it moves with the replicates, the spacing of the doses and the gate's level; a gene can pass the gate with no dose called, when its response spreads over several doses that each fall short. Under the 1.5-fold gate above, the lowest effective dose is the lowest dose whose change beyond 1.5-fold is called. On simulated series, the overall false discovery proportion of these calls was 0.033 to 0.057 (standard errors 0.006 to 0.011), and the lowest effective dose lay a median of 8.9 to 14 times above the gene's true benchmark dose of a 1.1-fold change ([Validation](../validation.md#validation-dose-response-calibration)).

Every dose called, not only the lowest, comes from {api}`stage_wise_adjusted_p_values` on the same tables, here 167 gene-dose calls:

```python
import numpy as np

from cytepool.stats.multiple_contrasts import stage_wise_adjusted_p_values

each = result.table("hepatocyte", "each_dose")
doses = np.unique(each["dose"])
p = np.asarray(each["p_value"]).reshape(doses.size, -1).T
spline = result.table("hepatocyte", "spline")
adjusted = stage_wise_adjusted_p_values(spline["p_value"], p, alpha=0.05)
calls = adjusted <= 0.05  # genes x doses
```

(dose-response-model-curves)=
## Curves of named models

What shape does each response take? A test says that a gene responds, and the spline's band shows the response, but reports of dose response name a curve: the models of BMDS, EPA's Benchmark Dose Software, which NTP's genomic dose-response approach and EPA's Transcriptomic Assessment Products fit [@ntp2018; @usepa2024; @usepa2026], drc's four-parameter log-logistic curve for EC50s [@ritz2015], or DRomics' curves that rise and then fall [@larras2018; @delignettemuller2023]. With `models`, {api}`dose_response` fits a set of them to every gene of each cell type, and {api}`fit_dose_curves` to a matrix of values outside a cube:

```python
result = cp.dose_response(cube, cp.Design(cube.samples, "~ 1"), dose=axis, models="epa")
curves = result.table("hepatocyte", "curves")
best = curves.filter(curves["best"])
ax = plot_dose_response(result, "hepatocyte", genes=genes, model="best")
```

The printout gains one line per cell type, such as `curves (constant variance): best linear 342, exponential_2 328, exponential_4 92, hill 40, exponential_5 9, power 6` for the hepatocytes' 817 genes: the model of lowest AICc, Akaike's criterion corrected for small samples, of each gene. Most genes do not respond, and their best model is a line or exponential 2, the models of fewest parameters. {api}`DoseResponseResult.curve_fits` gives a cell type's fits as a {api}`DoseCurveFits`, with the dose-group means and covariance they were fitted to, and {api}`DoseResponseResult.curve` the curve of a model at any dose, as a change from the vehicle's mean.

(dose-curve-sets)=
### The models of each set

A set is one call's models, so that a model's name means one curve with one set of parameters and bounds there. x is the dose, D the top dose, d_min the lowest positive dose, and G the largest gap between adjacent doses, on ln dose among the positive doses for the log-Gauss-probit forms; Φ is the standard normal distribution function.

| Set | Model | Mean | Bounds | Dose groups needed |
| --- | --- | --- | --- | --- |
| `"epa"` | `linear` | g + b1 x | none | 3 |
| | `polynomial_2` | g + b1 x + b2 x^2 | b1 and b2 of one sign | 4 |
| | `power` | g + v x^n | 1 ≤ n ≤ 18 | 4 |
| | `hill` | g + v x^n / (k^n + x^n) | 0 ≤ k ≤ 5 D, 1 ≤ n ≤ 18 | 5 |
| | `exponential_2` | a exp(b x) | a ≥ 0, -100 / D ≤ b ≤ 100 / D | 3 |
| | `exponential_3` | a exp(sign(b) (abs(b) x)^d) | as `exponential_2`, 1 ≤ d ≤ 18 | 4 |
| | `exponential_4` | a (c - (c - 1) exp(-b x)) | a ≥ 0, 0 ≤ b ≤ 100 / D, 0 ≤ c ≤ 1,000 | 4 |
| | `exponential_5` | a (c - (c - 1) exp(-(b x)^d)) | as `exponential_4`, 1 ≤ d ≤ 18 | 5 |
| `"ll4"` | `ll4` | c + (d - c) / (1 + (x/e)^b) | 0.1 ≤ b ≤ 18, d_min / 100 ≤ e ≤ 5 D | 5 |
| `"dromics"` | `linear` | d + b x | none | 3 |
| | `exponential` | d + b (exp(x/e) - 1) | D / 100 ≤ abs(e) ≤ 10,000 D | 4 |
| | `ll4` | as in `"ll4"` | as in `"ll4"` | 5 |
| | `gauss_probit` | f exp(-z^2/2) + d + (c - d) Φ(z), z = (x - e) / b | 0 ≤ e ≤ D, G / 3 ≤ b ≤ 10 D | 6 |
| | `gauss_probit_monotone`, `gauss_probit_symmetric` | the same with f = 0, or with c = d | as `gauss_probit` | 5 |
| | `log_gauss_probit` | as `gauss_probit`, with z = (ln x - ln e) / b and d at dose 0 | d_min ≤ e ≤ D, G / 3 ≤ b ≤ 10 ln(D / d_min) | 6 |
| | `log_gauss_probit_monotone`, `log_gauss_probit_symmetric` | f = 0, or c = d | as `log_gauss_probit` | 5 |

`"epa"` holds the eight models of NTP's approach and ETAP in BMDS's forms and with BMDS's letters, and its bounds are BMDS's, written in dose units where BMDS states them on doses divided by the top dose, with one of CytePool's own: c, which BMDS leaves unbounded above for a rising curve, at most 1,000 ([Limits and differences](#dose-response-limits)). Hill's k is the EC50, the dose of half the change v. The exponential models' a and c are on the values' own scale: a is the level at dose 0 and c the ratio of the plateau's level to a's, so, unlike the additive models, their fits change with where 0 lies on that scale, as log2 CPM puts it at 1 count per million. Two choices differ from a single BMDS run: the quadratic's coefficients share a sign, pybmds's default restriction, and the better of both signs is kept, as are both directions of the exponential models, where BMDS fits the direction it detects in the data. `"ll4"` is drc's LL.4, the Hill curve with a slope from 0.1; Hill, sigmoid Emax and four-parameter log-logistic curves are one family [@kappenberg2023]. Its letters are DRomics': b > 0 the slope, d the level at dose 0, c the level at high doses and e the EC50, the dose of half the change. drc reports a rising curve with b < 0 and c at dose 0, whose estimates (b, c, d, e) are CytePool's (-b, d, c, e). `"dromics"` holds DRomics' families, each four- and five-parameter Gauss-probit form a model of its own. Below a slope of 0.1 a log-logistic curve changes by about a fifth of its range over 4 decades of dose, a line in log dose that the slope and the plateaus cannot tell apart.

A Gauss-probit curve's peak, or its inflection, lies within the doses, and its width is at least a third of the largest gap between adjacent doses: every dose of the range then lies within half a gap of a dose, where a bell keeps exp(-9/8) = 32% of its height or more. Without these bounds, a bell narrower than a gap can sit between two doses with a height of hundreds of log2 units that only its tails reach, and the likelihood rises along such a valley without reaching a maximum.

A model of p parameters is fitted only with a spare dose group, K - p of at least 1, and the others are listed in `models_not_fitted`: a curve through every dose group's mean leaves nothing to test its shape against, and in simulations of four dose groups, non-protective lower bounds of the benchmark dose reached 80% in some scenarios [@ringblom2014]. Exponential 2 to 5 are not fitted to a gene with a dose-group mean at or below 0, since their means are positive, and no model to a gene whose values do not vary, or whose means, in the weighted and moderated modes, have a variance of 0 to rounding; `not_fitted_reason` says why.

(dose-curve-variance)=
### Constant, weighted or moderated variance

Under `variance="constant"`, the default, each gene's values in the cell type's fit, voom's log2 CPM or limma-trend's, are refitted against dose with one variance for every sample, as BMDS, drc and DRomics fit a gene's samples. The dose-group means are sufficient: the samples' sum of squares about a curve f is the within-group sum of squares S_w plus Q = sum_k n_k (ybar_k - f(x_k))^2, so each model is fitted to the K means with weights n_k, and the log-likelihood is the samples', -n/2 (log(2π σ²) + 1) with σ² = (S_w + Q) / n.

Under `"weighted"`, a curve is fitted by generalised least squares to the dose-group means of the cell type's moderated fit, ybar = M (m_0, θ), where m_0 is the vehicle's mean at the samples' average of the design's other terms, θ each dose's difference from it and M = [[1, 0], [1, I]], with their covariance s² C, C = M V M'. V, the fit's unscaled covariance of (m_0, θ), carries voom's weights and the design's other terms, and s² is the gene's residual variance; `"moderated"` takes its moderated variance, on the residual plus prior degrees of freedom. The log-likelihood is that of the means, -K/2 log(2π) - 1/2 log det(s² C) - Q / (2 s²) with Q = (ybar - f)' C^-1 (ybar - f), whose curve part is the generalised least squares of generalised MCP-Mod [@pinheiro2014]. Precision then comes from the cell type's whole fit. Under `"~ 1"`, the residual variance has the same n - K degrees of freedom as the pooled within-group variance of constant variance, 18 at 9 doses x 3, which put an SD within 0.68 to 1.32 times the true SD in 95% of experiments, and it weighs each sample by voom's weights; the moderated variance adds the prior's degrees of freedom ([Empirical Bayes moderation](empirical_bayes.md)). Constant variance stays the default until simulations show that weighted fits keep the coverage of benchmark-dose bounds ([Potency and points of departure](potency.md#benchmark-dose-bounds)). A design with other terms than dose, such as `"~ sex"`, needs `"weighted"` or `"moderated"`: constant variance refits dose alone, so it refuses them.

(dose-curve-fitting)=
### How each fit is found

No fit needs a start or a seed. A model's parameters split into linear ones, its levels and heights, and at most two nonlinear ones, such as a location and a slope. At each point of a fixed grid of the nonlinear parameters the linear ones are solved exactly, by weighted least squares within their constraints, which variable projection makes possible [@golub1973]; the three lowest local minima of the grid each start a refinement that moves the nonlinear parameters within their bounds and solves the linear ones again at each step, and the best result is kept. The grids hold 48 locations evenly spaced on log scale, from d_min / 100 to 5 D for the Hill and log-logistic curves and from d_min to D for Gauss-probit peaks, with 0 where a model allows it; 24 exponents from 1 to 18 (64 for the power model); 64 rates of each sign from 10^-4 / D to 100 / D; and 24 slopes and widths across their bounds. A refinement stops when the decrease that a full Gauss-Newton step predicts falls below 10^-12 of the objective, and `converged` is False for the few fits that reach 500 iterations first, which keep the best point found. The line and the quadratic are solved in closed form. A program that starts each fit from its own guesses can stop below this maximum; the Validation page counts how often pybmds, drc and DRomics do.

(dose-curve-table)=
### Read the curves table

`table(g, "curves")`, built by {api}`DoseCurveFits.table`, has one row per gene and model fitted, each gene's models in set order, and the set's parameter columns, NaN where a model has no such parameter:

| Column | What it says |
| --- | --- |
| `passes_gate` | After the gene's ids, where the gate ran: the gene's call in the table `"gate"` ([Which genes count as responding](#dose-response-gate)), so that `curves.filter(curves["passes_gate"])` keeps the curves of the genes that pass it |
| `n_parameters`, `spare_groups` | p, and K - p |
| `log_likelihood`, `aic`, `aicc` | AIC = -2 LL + 2k and AICc = AIC + 2k(k + 1) / (n - k - 1), with k = p + 1 for the variance and n the samples [@hurvich1989], as DRomics counts them; AICc is infinite when n ≤ k + 1 |
| `akaike_weight`, `best` | exp(-Δ/2) over its sum, Δ the AICc minus the gene's lowest: each model's share of the gene's support [@burnham2002]; `best` marks the lowest AICc, the first in set order within 10^-8 |
| `lack_of_fit_p_value` | The F-test of the curve against free dose-group means, F = Q / ((K - p) s²) on K - p and `df` degrees of freedom, with s² the pooled within-group variance under constant variance, the pure-error test, and the fit's variance otherwise; exact for curves linear in their parameters; below 0.1 the curve misses the means [@usepa2012] |
| `max_abs_scaled_residual` | The largest abs(ybar_k - f(x_k)) / se_k over the doses; above 2 warrants a look [@usepa2012] |
| `parameters_at_bounds`, `converged`, `not_fitted_reason` | The parameters at a bound, whether the refinement converged, and why a model was not fitted to the gene |
| `turn_dose`, `turn_driven_by_dose` | Where a curve that rises and falls turns, and the dose group that drives the turn, 0 for the vehicle's group and NaN for none ([Find responses that turn](#dose-curve-turns)) |

Of the 88 genes that the spline calls in the hepatocytes, the best model's lack-of-fit p-value is below 0.1 for 22 and its largest scaled residual above 2 for 30: steps, Fréchet curves and bells lie outside the EPA models, and three replicates leave the shape uncertain. Of the 27 true Hill responses among them, 6 have the Hill model as their best and 14 exponential 4, whose rise to a plateau differs from the Hill curve mostly in how it approaches the plateau; the Akaike weights say how sure each choice is. `curve(g, genes, model="best")` gives each gene's best curve as a change from the vehicle's mean that it was fitted to, whose value at the vehicle shows how far the curve misses that mean.

BMDExpress, in which NTP and ETAP run these models, reports for each fit an AIC and a fit p-value, the likelihood ratio test that BMDS calls test 4 [@ntp2018]. Under constant variance, `aic` is that AIC with every parameter counted: BMDS's default leaves out a parameter at a bound, so its AIC is lower by 2 for each, such as a power curve's n at 1 or a Hill curve's n at 18. Test 4 compares the same curve with free dose-group means as the F-test, by the likelihood ratio n log(1 + (K - p) F / (n - K)) on a chi-square of K - p degrees of freedom, which rejects too often at a few replicates: for a correct straight line at 9 doses x 3, its p-value falls below 0.1 in 25% of experiments, where `lack_of_fit_p_value` does in 10%. `best` is the lowest AICc over every fit, whether it fits or not. BMDS takes the lowest AIC among adequate fits, those that pass its checks, when their BMDLs lie within 3-fold, and otherwise the lowest BMDL [@usepa2012; @usepa2026]; NTP and ETAP take the lowest AIC [@ntp2018; @usepa2024]. Until rule sets reproduce these choices, keep the rows with `lack_of_fit_p_value` of 0.1 or more and take each gene's lowest `aic` to come close.

(dose-curve-turns)=
### Find responses that turn

Which responses rise and then fall? Of the models, only the Gauss-probit curves with f other than 0 can turn within the doses, which `"dromics"` holds. Their slope, exp(-z²/2) (-f z + (c - d) / sqrt(2π)) / b, changes sign once, at z* = (c - d) / (f sqrt(2π)), so the turn is at x* = e + b z*, or e exp(b z*) on ln dose, and `turn_dose` gives it when it lies strictly between 0 and D: a curve that peaks at the top dose rises over every dose. A bell can also fit one outlying dose group, which EFSA found behind most apparent non-monotone responses [@efsa2021]: for each gene whose best fit turns, the whole set is fitted again without each dose group in turn, the vehicle's included, and `turn_driven_by_dose` gives the lowest dose without whose group the best fit by AICc no longer turns, 0 when it is the vehicle's group. The check needs 6 dose groups: at 5, no model that can turn has a spare group among the 4 left without one, so no refit could turn; `turn_driven_by_dose` is then NaN for every gene, and the printout and the record say that the turns were not checked.

```python
turning = cp.dose_response(
    cube,
    cp.Design(cube.samples, "~ 1"),
    dose=axis,
    models="dromics",
    groups="hepatocyte",
)
best = turning.table("hepatocyte", "curves")
best = best.filter(best["best"])
```

In the hepatocytes, 159 of the 817 genes have a best fit that turns. 120 of them are among the 655 flat genes, and one dose group drives the turns of 113 of these and of all 18 monotone responses whose best fit turns. Of the 33 true bells, 21 have a turning best fit, and 14 of these keep their turn without each group, as do 7 flat genes. So read turns among the genes that a test calls, such as those that pass the gate (`passes_gate`), and keep those that turn without each group: of the 88 genes that the spline calls, 27 have a turning best fit, and the 14 that keep their turn are all true bells.

(dose-curve-descriptive)=
### Describe cell types without replicates

A cell type that supports only description ([Checking a dose design](#dose-design-check)) gets curves under constant variance in every mode, from the values a fit would compute, the check's samples, the analysis's expression filter with the model of the dose groups as its design, TMM factors and voom's or limma-trend's log2 CPM, without lack-of-fit p-values: its within-group variance comes from the vehicle's replicates alone, if any. With other terms than dose in the design, such a cell type gets no curves, with the reason, since constant variance fits dose alone.

(dose-response-signatures)=
## Signatures along dose

Does a signature scored in each sample respond to the dose, in the direction expected? Pharmacodynamic trials, such as those in lupus, follow a predefined score: of 61 clinical studies that profiled people at two or more doses, the one that fitted a dose model ran MCP-Mod on a 22-gene type I interferon score of blood, measured by TaqMan assays and analysed as each patient's change from baseline, in a phase 2 trial of litifilimab [@furie2025], and EPA's high-throughput transcriptomics pipeline fits concentration-response curves to signature scores [@harrill2024b]. {api}`dose_response` takes the scores of one cell type's samples in place of a cube, the pathway scores of {api}`gsva`, {api}`ssgsea`, {api}`combined_z_scores` or {api}`plage`, or the activities of {api}`ulm`, {api}`mlm` or {api}`weighted_aggregate`, and every test, table and curve of this chapter runs on them. On the series of [Which genes respond to dose](#dose-response-tests), a signature of the 30 hepatocyte genes whose expression falls along Hill curves, and 30 flat genes as its control:

```python
import numpy as np

genes = truth.table("hepatocyte")
ids = np.array(genes["feature_id"].tolist())
change = np.asarray(genes["max_log2_change"])
falling = (np.array(genes["shape"].tolist()) == "hill") & (change < 0)
sets = {"falling": ids[falling].tolist(), "flat": ids[change == 0][:30].tolist()}
scores = cp.gsva(cube, sets, group="hepatocyte")
signatures = cp.dose_response(
    scores,
    cp.Design(cube.samples, "~ 1"),
    dose=axis,
    tests=("linear_trend", "mcp"),
    direction="down",
    rng=1,
)
trend, mcp = signatures.table(name="linear_trend"), signatures.table(name="mcp")
for i, name in enumerate(trend["gene_set"]):
    print(
        f"{name}: slope {trend['estimate'][i]:+.2f}, p {trend['p_value'][i]:.2g}; "
        f"MCP {mcp['contrast'][i]}, t {mcp['t'][i]:.1f}, p {mcp['p_value'][i]:.2g}"
    )
```

```text
falling: slope -0.29, p 9.5e-08; MCP emax_2, t -8.9, p 1e-07
flat: slope +0.02, p 0.7; MCP beta, t -1.0, p 0.44
```

The falling set's score drops by 0.29 GSVA units per log10 unit of dose, and the MCP step finds its fall most like `emax_2`, the Emax shape that reaches half its change at 2.1 ug/kg; the flat set shows no fall. Of the falling set's 30 genes, the 15 that pass the expression filter were scored, as `scores.n_genes` says. With two scores there is no moderation, and each p-value is that of the score's own linear model.

What differs from the tests of genes:

- **The fit.** A score is not a count, so it gets no voom weights: the scores are fitted as {api}`limma` fits them, by limma's linear model of the dose groups and empirical Bayes moderation across the scores, as GSVA's guide tests its scores [@haenzelmann2013; @smyth2004], with limma's options `trend` and `robust`. The design's samples are matched to the scores' profiles by `sample_id`; those without a score, such as the samples that {api}`gsva` left out for having too few cells, are left out, with the reason in `result.design_report.samples`, and a score with missing values is fitted on the samples it has. A score that does not vary beyond what the model of the dose groups fits, such as an assay at its floor, has residuals of rounding alone, which limma would test: it is fitted as a missing score, out of the moderation, with the reason in the fit's record (`scores_not_tested`). Infinite scores are refused.
- **How many scores.** limma does not moderate one or two scores, so a signature tested alone, or beside one control set, gets the tests of its own linear model, the model DoseFinding's `MCTtest` fits: on null series, a GSVA score tested alone had p-values below 0.05 in 3.9% to 7.3% of series. Moderate 30 sets or more with `trend=scores.n_genes`, which lets the prior variance follow the set sizes, or `trend=activities.n_targets.mean(axis=0)` for sources: then 4.6% to 6.2% of the null sets in each third by size had p-values below 0.05. With fewer sets, test each set alone, one call per set on the scores of `gsva(cube, {name: genes}, group=...)`, and adjust the sets' p-values together with {api}`adjust_p_values` ([Validation](../validation.md#validation-dose-response-signatures) gives the rates of every design).
- **The tables.** One row per gene set or source, with the columns {api}`limma` gives scores: `gene_set` or `source`, `estimate`, the change in the scores' units (per log10 unit of dose for the linear trend), and `average`, the mean score over the samples fitted. `curve()`, the curves of named models and the lowest effective doses name them alike.
- **One cell type per call.** Scores belong to the cell type they were computed in, and `groups`, `min_cells` and `min_counts`, which choose a cube's samples, stay at their defaults; scores of an array form the cell type `"bulk"`. {api}`DoseResponseResult.fit` computes a cube's fit again and refuses scores, whose tables and curves come from the fit the result keeps. The [gate](#dose-response-gate) takes no fold change, which a score does not have, and {api}`permutation_check`, which runs a cube's analysis again, refuses a result of scores.
- **The sign of a score.** A direction means what the score's sign means. GSVA's default score, ssGSEA's and the combined z-scores rise when the set's genes rise; CytePool's PLAGE score rises with the set's mean standardized expression, where GSVA's PLAGE score takes an arbitrary sign; a ULM activity rises with the source's activity. GSVA's Kuiper score (`score_type="kuiper"`) is at least 0 and rises when the set's genes move either way, so test it two-sided, or with `direction="up"` for a change of its genes either way.
- **Ranks within a sample.** GSVA and ssGSEA rank each sample's genes against one another, so a broad response of other genes moves a set's score: a flat set scored beside the signature, as above, shows how far. GSVA and the z-scores also set each gene against the samples scored, so score the series' samples together.
- **Curves of scores.** The exponential forms of `"epa"` need positive means, which scores centred on 0 rarely have, so those fits are declined; `"ll4"` and `"dromics"` fit scores of any sign.
- **Scores of other assays.** A score measured by another assay, such as a TaqMan panel, enters through its genes' values: {api}`combined_z_scores` or {api}`gsva` of an array of samples x genes, with `feature_ids` and `sample_ids`, gives the scores of the cell type `"bulk"`. A score given as one value per sample, without its genes, is not taken yet.

(one-sided-dose-tests)=
## Tests in one direction

Did the response go the way it was expected to? A two-sided test finds a change either way. A one-sided test spends its whole error rate on one direction, so it finds a change that way with fewer samples. `direction="up"` or `"down"`, DoseFinding's `"increasing"` and `"decreasing"`, makes the tests with a sign one-sided, for genes and scores alike, and each test calls a change of its own:

| Test | `"up"` | `"down"` |
| --- | --- | --- |
| `linear_trend` | A positive slope along the axis, through the treated doses when the vehicle is a group of its own | A negative slope |
| `dunnett` | A dose above the vehicle | A dose below the vehicle |
| `williams` | The pooled top doses above the vehicle | The pooled top doses below the vehicle |
| `mcp` | Agreement with one of the candidate shapes, as they rise | Agreement with one of their mirror images, as DoseFinding's `Mods(direction = "decreasing")` gives them |

For one t-statistic, the one-sided p-value is half the two-sided one when t points the stated way, and 1 minus half of it otherwise. The multiple contrast tests refer the largest of their t-statistics in the stated direction to the multivariate t distribution [@hothorn2008]: their contrast need not be the one of largest \|t\|, so their p-values are not halves of the two-sided ones. Dunnett's and the Williams-type tests compare doses with the vehicle, so a test for a fall calls no response that stays above the vehicle. The MCP step's contrasts and the linear trend compare the dose groups among themselves, so for a fall they can call a score that never falls below the vehicle: the MCP step a rise at the top doses, which follows the mirror image of the umbrella shape, a U, and both a rise in the middle doses or at the lowest dose that returns, which follows the mirror images of the exponential and Emax shapes and slopes down through the treated doses. `CandidateShapes.spread(dose, design.samples, n_beta=0)` leaves the umbrella out of the MCP step's candidates. Standard errors, t-statistics and `log_odds`, limma's B-statistic, stay those of the two-sided tests. A gene or score that moves the other way gets a p-value near 1, often exactly 1, with a bound on its distance from 1 as its `p_value_error` ([Multivariate t probabilities](multiple_testing.md#multivariate-t-probabilities)).

The F-tests `spline`, `nonlinearity` and `dose_groups` test a change in any direction and have no one-sided form in limma's model, so `direction` refuses them, and the default tests with them: name the tests, as `tests=("linear_trend", "mcp")` does. Each dose against the vehicle, the curves and their bands stay two-sided. The [gate](#dose-response-gate) takes the direction too: its test is one-sided, the Williams-type test by default, which then runs as well, since it compares the top doses with the vehicle, or Dunnett's or the MCP test when the gate names them, and a gate of an F-test or a fold change is refused. A one-sided gate still tests, at its level, that every dose equals the vehicle, the hypothesis that the closed test behind the lowest effective doses needs, and each dose's call stays two-sided, so a dose called after a gate for a fall can be a rise.

DoseFinding's `MCTtest` is one-sided by default, at a level of 0.025, and reports positive t-statistics for decreasing models, where CytePool reports negative ones of its rising contrasts. At the gate's default FDR of 0.05, a one-sided test is about as lenient in its direction as a two-sided test at 0.10, and exactly so for one t-statistic; `gate={"fdr": 0.025}` calls at `MCTtest`'s level. Use a direction fixed before the data, such as a drug's expected effect on its pharmacodynamic signature: a direction chosen after looking halves a p-value without new evidence.

(simulated-dose-series)=
## Simulated dose series with known curves

How well does a dose method find the genes that respond, the shape of each response and the dose at which it starts, at a given design? Only data whose truth is known can say, and the one single-cell dose benchmark compared tests on cells, without curves [@nault2022]. {api}`simulate_dose_series` draws a cube of pseudobulk counts from known curves of each gene's expression along dose, and {api}`DoseTruth` holds the truth. Here the spline F-test of [Which genes respond to dose](#dose-response-tests) runs on a simulated series, and the truth says which genes it should call:

```python
import numpy as np

cube, truth = cp.simulate_dose_series([0, 0.01, 0.1, 1, 10, 30], replicates=3, rng=1)
axis = cp.DoseAxis("dose", unit=truth.unit, metric=truth.metric)
result = cp.dose_response(cube, cp.Design(cube.samples, "~ 1"), dose=axis)

tested = result.table("cell_type_1", "spline")
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
# 62 of 113 responsive genes called, 3 of 461 flat
# maximal change 0.5: 4 of 32 called
# maximal change 1: 24 of 38 called
# maximal change 2: 34 of 43 called
```

Of the 574 genes that pass the expression filter, the spline's F-test calls 62 of the 113 that respond and 3 of the 461 flat ones at 5% FDR: with 3 replicates per dose and 100 cells per profile, nearly half of the responses go unseen while false calls stay rare. 28 of the 51 misses are changes of 0.5 log2 units, of which 4 in 32 are called, against 34 in 43 changes of 2. The same lines with `replicates=4` or `cells=30` show what another design gains or loses before it is run.

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
- **The tests.** Every test the design supports, and the gate's test; in a cell type whose dose groups are few, one of two tests of the same hypothesis, the gate's test kept ([Which genes respond to dose](#dose-response-tests)). Dunnett's, the Williams-type and the MCP tests run only when named, or when the gate takes one, two-sided ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts)).
- **The gate.** The spline test at 5% FDR, Benjamini-Hochberg within each cell type; the dose-group test when fewer than two doses are on the axis ([Which genes count as responding](#dose-response-gate)).
- **Candidate shapes.** Five, spread over the design's doses by {api}`CandidateShapes.spread`: two Emax shapes, a sigmoid Emax shape of Hill slope 3, an exponential shape and a beta shape peaking mid-range.
- **The permutation check.** 100 runs per cell type, or every other assignment when there are no more, from a seed drawn and recorded unless `rng` gives one; each run fits the gate's test alone, with the result's seed ([Could the calls come from noise?](#dose-permutation-check)).
- **The lowest effective dose.** Each dose of a gene that the gate passes, called by Holm's step-down procedure with Shaffer's first step at the level the gate leaves ([From which dose a gene responds](#lowest-effective-dose)).
- **The analysis.** {api}`limma_voom` with exact standard errors, its expression filter and TMM normalization, on the model of the dose groups.
- **The spline.** 2 columns when the axis holds 4 doses or more, otherwise 1, in the restricted cubic form, with knots at quantiles of the distinct doses of the design's samples.
- **FDR.** Benjamini-Hochberg within each cell type and test, and across the cell types that ran a test. With 2 replicates per dose, a cell type in which no gene responds has a call more often than 1 time in 20; {api}`permutation_check` tests whether a series' true doses pass more genes than permuted ones ([Limits and differences](#dose-response-limits)).
- **Curves.** None; with `models`, constant variance, as BMDS fits a gene's samples, the best model by AICc, and the F-test of lack of fit read at 0.1 ([Curves of named models](#dose-response-model-curves)).
- **Direction.** Two-sided tests; `direction="up"` or `"down"` for the tests with a sign, with the Williams-type test as the gate ([Tests in one direction](#one-sided-dose-tests)).
- **Scores.** Fitted as {api}`limma` fits them, moderated across the scores given, with a constant prior variance (`trend=False`) as in limma ([Signatures along dose](#dose-response-signatures)).
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
- **Descriptive cell types.** {api}`dose_response` lists them with the check's reason and gives them no tests or estimates; with `models`, they get curves without lack-of-fit p-values ([Describe cell types without replicates](#dose-curve-descriptive)).
- **Each dose against the vehicle.** The table `each_dose` adjusts each dose's tests within the cell type and dose, without control over the several doses of a gene, which {api}`multiple_contrast_test` on the cell type's fit gives dose by dose and `"dunnett"` for the gene as a whole ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts)), or across genes, which {api}`DoseResponseResult.lowest_effective_doses` gives.
- **Degrees of freedom that are not integers.** The multiple contrast tests take a gene's moderated degrees of freedom as they are, where DoseFinding's `MCTtest` and multcomp's `glht` refuse non-integer ones ([Validation](#validation-dose-contrasts)).
- **Degrees of freedom under dream.** Under {api}`dream_voom` each contrast has its own degrees of freedom, and the multivariate t takes the smallest of the gene's doses', an approximation. It is conservative for Dunnett's test, whose contrasts are the doses. In a simulated design of 4 donors and 5 dose groups with 5 of its 20 samples left out, the Williams-type test's contrasts had more degrees of freedom than any dose, but the MCP step's had fewer in 19 to 174 of 240 genes, depending on the shape, down to 0.83 times the smallest dose's.
- **One set of contrasts per cell type.** The Williams-type and MCP contrasts follow a cell type's samples per dose group, as the spline does, not each gene's precision; each gene's covariance, voom's weights included, still sets the contrasts' standard errors and correlation.
- **Holm within each gene.** The per-dose calls of {api}`DoseResponseResult.lowest_effective_doses` leave out the correlation of a gene's doses through their shared vehicle, which the single-step max-t test uses. The gain it gives up is large with few degrees of freedom and small with many: at single-step p-values of 1e-6, Bonferroni's p-values are 22% larger for 3 doses and 65% for 8 at 5 degrees of freedom, and 1% and 3% at 40 ([Several contrasts of one gene](multiple_testing.md#correlated-tests)).
- **Responses with no dose called.** A gene that the gate passes can have no dose called, when its change spreads over several doses that each fall short; the lowest effective dose is a tested dose, not a potency between doses.
- **Non-monotone responses.** The Williams-type test assumes the response grows in dose order and misses many bells; the spline and the MCP step detect most of them, the MCP step partly through its beta shape: many bells turn back only at the top doses, where they look monotone.
- **The spline's weights.** The spline passes through the dose groups' means weighted by their samples, not by each gene's precision, so under voom's weights a dose group of few cells counts as much as its samples, though its mean is less precise.
- **Bands where the spline cannot follow.** A band holds the spline fitted to the dose groups' true means, which misses a response that jumps or turns more sharply than the spline can follow; read the band beside each dose's estimate ([Curves with a band](#dose-response-curves)).
- **Curves of `dream_voom` fits.** Not implemented yet: constant variance would leave out the random intercepts, and the weighted and moderated modes need the degrees of freedom of contrasts that dream's fits do not keep.
- **AICc of the weighted and moderated modes.** Its correction counts the variance as estimated from the cell type's samples, as under constant variance, though the curve takes it from the cell type's fit.
- **Weighted curves under limma's standard errors.** With `contrast_standard_errors="limma"`, the weighted and moderated modes take limma's approximate covariance of the dose groups' estimates, as its tests do.
- **Levels far beyond the data.** A Hill or log-logistic curve whose midpoint lies at its bound beyond the top dose, or whose slope is at 18, can fit a change at the top dose with a level far beyond the data: in the hepatocytes of the example, 83 of the 817 Hill fits and 82 of the 817 log-logistic fits have a level beyond 100 log2 units, up to 5.8e12, none of them a gene's best fit. Each has a parameter at a bound, so read the levels, k and e only where `parameters_at_bounds` is empty, and the others by their curve.
- **The bound on c.** BMDS sets no upper bound on c of a rising exponential 4 or 5 curve. CytePool bounds it at 1,000, which keeps finite the fits that approach a straight line, where c grows as b shrinks, and lies far beyond the plateaus of log2 values: c is the ratio of the plateau's log2 level to the vehicle's, not a fold change, so c = 1,000 is a plateau of 20 log2 CPM over a vehicle at 0.02. In the hepatocytes, 69 of the 817 exponential 4 fits stop at c = 1,000, fits that approach a line.
- **DRomics' selection.** Its filters, a fit's AICc against the flat model's and checks of the residuals and of extrema beyond the doses, are not applied: each model's fit is reported, and the gate ([Which genes count as responding](#dose-response-gate)) decides which genes respond.
- **Two replicates per dose.** In series where no gene responds, a cell type of 1,000 genes had a call at 5% FDR in 6.4% to 18.3% of the datasets with 2 replicates per dose, depending on the test and the number of doses, and in 4.5% to 7.5% with 3, the multiple contrast tests included, where Benjamini-Hochberg aims at 5% ([Validation](../validation.md#validation-dose-response-calibration)). The default gate passed a gene in 15.5% and 10.2% of such cell types at 4 and 9 dose groups with 2 replicates, beyond the 6.95% that 2,000 cell types allow at 5%, and in 6.9% and 6.8% with 3; the dose-group gate in 7.5% and 7.3% with 3, beyond it too; the 1.5-fold gate in at most 1.1%. With few residual degrees of freedom, 4 for the vehicle and three doses in duplicate, moderation pulls each gene's variance most of the way to a common prior, which understates the variance of genes that vary more than most between replicates. At 2 replicates per dose, read a few calls in a cell type as possibly due to chance, and run {api}`permutation_check`, whose p-value tests whether the true doses pass more genes than permuted ones ([Could the calls come from noise?](#dose-permutation-check)).
- **Designs the check cannot permute.** {api}`permutation_check` refuses a design with more than one random effect or a random slope, whose exchangeable units it does not work out, and lists as not checked a cell type in which every sample is a block of its own, as a numeric covariate of the design makes it: drop the covariate, or bin it into a factor. The runs leave out `batch`, which a permuted labelling could confound by chance and which no fit models.
- **Doses that change the cells.** The runs keep each sample's cells and library size. A dose that changes how many cells a sample holds changes the precision of its profile too, so the samples of different doses are not exchangeable even where no gene responds, and the check's share and p-value are then approximate.
- **Genes that respond, in the runs.** The runs keep every gene's response, scrambled over the permuted dose groups, and a strong response still passes in a run that leaves most of a dose group's samples together: in the hepatocytes of [Dose response per cell type](../guides/dose_response.ipynb), 47 of the 51 genes that passed in the runs respond in truth. The p-value stays a valid test of whether any gene responds, and comes out larger than noise alone would make it. On simulated series in which a fifth of 1,000 genes respond, the median share of runs that passed genes was 0.06 to 0.20 across four designs, against 0.07 to 0.16 of series in which no gene responds, and every check gave a p-value at most 0.05 ([Validation](../validation.md#validation-dose-response-calibration)).
- **A fold change under dream.** A gate with a fold change refuses fits of {api}`dream_voom`: TREAT takes limma's moderated t-tests, which dream's tests are not.
- **What the runs repeat.** Each run repeats the analysis up to the gate: the fit and the gate's test alone. Curves and the other tests do not run, since the gate does not read them, and a gate of a multiple contrast test costs its test in every run ([Could the calls come from noise?](#dose-permutation-check) gives the times).
- **limma's standard errors.** With `contrast_standard_errors="limma"`, the F-tests of the spline use limma's approximate standard errors, which depend on the spline's columns: the F-statistics of R's `ns()` columns and of the restricted cubic columns of the same spline differ by up to half ([Validation](#validation-dose-response-tests)). The exact default and {api}`limma_trend` do not depend on them. The multiple contrast tests refuse limma's standard errors, as {api}`multiple_contrast_test` does: they give no covariance of a gene's doses' estimates, and on a simulated series of 9 dose groups the Williams-type and MCP p-values moved by up to 100-fold with them.
- **Curves of dream fits and beyond the doses.** {api}`DoseResponseResult.curve` gives no curve for results of {api}`dream_voom`, whose tests run in full, and no value outside the doses a cell type compares.
- **Scores that follow subtypes.** A score can move with a shift between the subtypes of a cell type, as when a dose favours an activated state, rather than with a change within each cell: a composition test of the series, {api}`propeller` with dose in the design, shows such a shift ([Composition tests](composition.md)).
- **Scores of few cells.** Scores carry no precision weights, so the score of a profile of few cells, noisier than the others, weighs as much in the fit.
- **Few moderated scores.** limma estimates the prior variance from the scores' own variances and then treats it as known, which gives null sets p-values below 0.05 too often when 3 to 10 are moderated together; [Signatures along dose](#dose-response-signatures) says when to moderate.
- **Designs of scores.** No random intercepts or sample weights: pool the samples of each donor at each dose with {api}`Cube.collapse_samples` before scoring.
- **What a simulated series leaves out.** In {api}`simulate_dose_series`, a gene's biological SD is the same at every dose, the constant variance that BMDS fits by default; genes are independent given their curves, so gene-set methods meet no correlation between genes; each cell type draws its responsive genes on its own, so few genes respond in several cell types; bells are symmetric, and no curve is hormetic; the counts hold no ambient RNA; and library sizes move with dose, by a median of 0.08 log2 at the top dose of default series, which TMM removes for the most part, leaving about 0.01 log2 on the flat genes at the top dose.

(dose-response-checks)=
## Checks and further reading

Dose axes, dose designs, dose tests, curves of named models and simulated dose series are compared with R, limma, scanpy, pybmds, drc, DRomics, scipy, mpmath and published data, run as separate programs on the same inputs:

- {api}`DoseAxis`: positions, knots, columns, designs, F-tests and fitted curves against R 4.4.3's `ns()`, Hmisc's `rcspline.eval`, `model.matrix`, `lm` and `predict`, exact (1e-10); moderated F-tests of `limma_trend` against limma 3.62.1's, exact on `ns()`'s columns and close (1e-6) on Hmisc's, with every column counted, and equal in the two forms ([Validation](#validation-dose-axes)).
- {api}`check_dose_design`: cells, counts and the samples used per dose group against scanpy 1.12.4 and pandas; the residual degrees of freedom, estimable dose groups and groups confounded with a batch against R 4.4.3's `qr()` and limma 3.62.1's `nonEstimable`; statuses and ETAP's minimums against the same rules coded in R; and GSE208161, where every cluster is descriptive, all identical ([Validation](#validation-dose-design)).
- {api}`simulate_dose_series` and {api}`DoseTruth`: true curves, EC50s and benchmark doses against mpmath 1.3.0, exact (1e-10); the means and variances of cells and counts within 4 standard errors of their exact values; the same series at 1, 2 and 8 threads ([Validation](#validation-simulate-dose-series)).
- {api}`dose_response`: the samples, dose groups and genes of each cell type, its F-tests, slopes, each dose against the vehicle, adjusted p-values and curves with their bands against limma 3.62.1 and edgeR 4.4.0 run on R's own knots and spline columns, the exact default against each gene's weighted least squares in R, exact (1e-10); the spline's sums of squares against rms 8.2-1, exact; the dream route against variancePartition 1.36.2, close (1e-6); DRomics, dreamlet and muscat on the same profiles, as information ([Validation](#validation-dose-response-tests)); error rates under the null, power and the coverage of the bands on simulated series ([Validation](#validation-dose-response-calibration)).
- {api}`fit_dose_curves` and the curves of {api}`dose_response`: each fit's maximum against pybmds 25.2, drc 4.0-0's LL.4 and a 200-start scipy 1.18.1 search on the same data, never lower by more than 1e-8 of the residual variance, and against DRomics 2.6.3's `drcfit`, never higher in AICc by more than 0.005, the rounding of its printed criteria, with the parameters where both reach the maximum; the AIC's penalty, and test 4 from the F-test, against pybmds; the F-test of lack of fit against R's `anova` under constant variance and limma 3.62.1 otherwise, with the means, covariance and variances of the weighted and moderated modes, exact; and whether each best fit turns, and the dose group that drives a turn, against the rules coded apart ([Validation](#validation-dose-curves)).
- {api}`plot_dose_response`: the drawn curves, bands and points against limma-trend's spline in R, and the curves of a model against drc's log-logistic formula evaluated in R, exact ([Validation](#validation-plotting)).
- {api}`dose_response` on pathway scores, and `direction`: the samples fitted, every test's statistics and p-values, and the curves and bands against limma 3.62.1 with R's own knots and spline columns, exact (1e-10), close (1e-6) with `robust=True`; the one-sided linear trend against `pt`, exact; the MCP step's contrasts and t-statistics on one score against DoseFinding 1.4-2's `optContr` and `MCTtest`, Dunnett's and Williams-type t against multcomp 1.4-32, exact, and their p-values in both directions within simulation error of 10 million draws of the multivariate t; curves of one score against DoseFinding's `fitMod`, close; activities, which take the same path, by a unit test of identical statistics; and the check of scores against the check of their cube ([Validation](#validation-dose-response-signatures)).
- {api}`CandidateShapes`, {api}`optimal_contrasts`, the three multiple contrast tests, {api}`stage_wise_adjusted_p_values` and {api}`DoseResponseResult.lowest_effective_doses`: shapes against DoseFinding 1.4-2's `Mods`, `getResp` and `guesst`; contrasts against `optContr` and `MCTtest`; the tests' statistics against limma 3.62.1 and variancePartition 1.36.2 fits in R with multcomp 1.4-32, exact (1e-10), close (1e-6) for dream; their p-values against mvtnorm 1.4-2 and scipy 1.18.1 within simulation error; stage-wise p-values and lowest effective doses against stageR 1.28.0 and a closed test enumerated in R, exact and identical ([Validation](#validation-dose-contrasts)); error rates and power on simulated series ([Validation](#validation-dose-response-calibration)).
- The gate of {api}`dose_response` and {api}`permutation_check`: the dose-group gate's p-values against limma 3.62.1 from the counts, under voom's weights and with `classifyTestsF`, exact (1e-10) and close (1e-6) with robust priors; each dose's TREAT p-values against `treat` and `topTreat`, exact; the genes passed, and the lowest effective doses under a fold-change gate against stageR 1.28.0, identical; the units, the assignments that R counts and enumerates, and the genes passed in every run and the ties of designs of 90 and 36 assignments, refitted by limma in R, identical; runs under dream against variancePartition 1.36.2, close ([Validation](#validation-dose-gate)); the share of null cell types that each gate passes, and the check's runs against it, on simulated series ([Validation](#validation-dose-response-calibration)).

Further reading:

- @desquilbet2010 explain dose-response analyses with restricted cubic splines, their knots and their tests of any response and of departure from a line.
- @ntp2018 sets out the National Toxicology Program's approach to genomic dose-response modelling, from its ANOVA gate to benchmark doses.
- @lun2017 showed that plate effects confounded with the condition break the control of false positives, and that summing the counts within each plate restores it.
- @crowell2020 benchmarked differential state methods on simulated and real multi-sample data, and found pseudobulk methods performing decently from about 100 cells per cell type and sample.
- @usepa2024 sets out the agency's standard methods for transcriptomic assessment products, with the design minimums of Sections 3.4.1.3, 3.4.2 and 3.4.5.2.
- @ema2014 qualified MCP-Mod for phase 2 dose finding, with its minimum numbers of doses; @bretz2005 introduced the combination of multiple comparisons and modelling behind it, and @pinheiro2014 its generalization to estimates and their covariance from any model; @duda2022 applied it gene by gene to expression data.
- @ich1994 sets out the dose-response information that supports drug registration, including the effect of the lowest recommended dose; @heller2009 and @vandenberge2017 describe the stage-wise testing behind the lowest effective doses.
- @hurvich1989 derived the AICc for small samples, and @burnham2002 explain Akaike weights; @larras2018 introduced DRomics' models and their selection; @ringblom2014 showed how often benchmark doses of curves without a spare dose group mislead; @efsa2021 weighs the evidence for non-monotone dose responses.
- @furie2025 analysed a 22-gene interferon score along the doses of a phase 2 trial by MCP-Mod, and @harrill2024b fit concentration-response curves to signature scores of high-throughput transcriptomics screens.
- @nault2022 benchmarked tests of single-cell dose series simulated from the EPA's benchmark-dose equations; @crowell2020 and @song2024 describe the simulators of muscat and scDesign3; @crump1984 introduced the benchmark dose, and @kappenberg2023 reviews the design and analysis of 5,670 toxicological dose-response analyses.
- @bourgon2010 showed that a filter before a test keeps the test's control of errors when the filter's statistic and the test's are independent under the null hypothesis, and that some filters in use lose it; @mccarthy2009 introduced TREAT, which tests changes against a fold-change threshold instead of filtering on the estimated changes.
- @lee2024 measured how often a benchmark-dose workflow gives a point of departure from vehicle-only zebrafish embryos placed in mock dose groups, and how pooling embryos lowers it; @bruns2026 compared five workflows on one dose series, which reported 70 to 3,618 responsive genes at 4 h, and ran them on permuted concentration labels; @phipson2010 showed why permutation p-values should never be zero, and how to compute exact ones when the permutations are drawn at random.

[Potency and points of departure](potency.md) reads from each curve the dose at which the response reaches a stated change, its benchmark dose, with bounds.
