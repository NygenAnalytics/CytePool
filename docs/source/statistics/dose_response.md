---
description: Where doses sit on a dose axis, what a dose series supports, which genes respond to dose in each cell type, with curves, trends and candidate shapes, from which dose, and simulated dose series.
---

# Dose response

[Composition tests](composition.md) asked whether a condition changes which cells a sample holds. A dose series asks of each cell type which genes respond to a dose and how, and two things come before any test or curve: where each dose sits on the axis of a curve, which {api}`DoseAxis` records, and whether the samples of each cell type can support tests at all, which {api}`check_dose_design` reports. {api}`dose_response` then tests every cell type that supports tests, with a fitted curve and its band for each gene, tests of trends, of each dose against the vehicle and of candidate shapes of response, and the lowest dose from which each gene responds, and series simulated from known curves with {api}`simulate_dose_series` show how well a dose method recovers the responses at a design before it is run. [Dose response per cell type](../guides/dose_response.ipynb) runs the steps on a simulated series.

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

Read `spline` first: it asks whether a gene responds at all. The linear trend and the non-linearity test describe the responses it finds, and `dose_groups` asks the same question as `spline` without a curve. The dose-group test is the one-way analysis of variance of the dose groups with moderated variances, the moderated counterpart of the ANOVA that the genomic dose-response approach of the National Toxicology Program runs as a gate: at least one gene must pass it at a Benjamini-Hochberg FDR below 0.05 for a data set to be analysed further [@ntp2018].

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
responding = spline.filter(spline["adjusted_p_value"] < 0.05)
```

```text
DoseResponseResult (limma_voom): 3 cell types, 2 tested; tests: spline (read
    first), nonlinearity, linear_trend, dose_groups
  dose: ug/kg (administered) on a log10 axis, the vehicle in a group of its own;
        spline of 2 columns, knots at 0.548 ug/kg, boundary knots at 0.01 and 30
        ug/kg; linear_trend slopes per log10 unit of dose
  hepatocyte: 9 dose groups, 27 samples, 817 genes; calls at 5% FDR: spline 88,
              nonlinearity 31, linear_trend 81, dose_groups 77
  stellate: 8 dose groups, 24 samples, 494 genes; calls at 5% FDR: spline 43,
            nonlinearity 9, linear_trend 38, dose_groups 34
  cholangiocyte: not tested, refused: no vehicle sample is left (3 vehicle
                 samples left out); compare the doses with each other with
                 limma_voom
```

The {api}`DoseResponseResult` prints the axis, the spline's knots in doses and the unit of the linear trend's slope, and for each cell type the dose groups it compares, its samples, the genes it tests and the genes each test calls at 5% FDR. The hepatocytes compare all 9 dose groups and the spline calls 88 genes, 85 of which respond in truth. The stellate cells keep no sample at 30 ug/kg, so they compare 8 dose groups, and their dose-group test has one degree of freedom fewer. The cholangiocytes keep no vehicle sample of 10 cells or more, so the check refuses them, and the summary gives its reason.

The result's tables, which {api}`DoseResponseResult.table` gives by cell type and name, and its summary:

| Table | One row per | What to look at |
| --- | --- | --- |
| `table(g, "spline")`, `"nonlinearity"`, `"dose_groups"` | Gene | `f_statistic`, `p_value`, `adjusted_p_value` and `adjusted_p_value_across_groups`; `df`, the denominator degrees of freedom |
| `table(g, "linear_trend")` | Gene | `log2_fold_change`, the slope; `t`, `p_value` and the adjusted p-values |
| `table(g, "each_dose")` | Gene and treated dose | Each dose against the vehicle: `log2_fold_change`, `standard_error`, `t`, `p_value` and `adjusted_p_value`, adjusted within the dose |
| `result.summary` | Cell type | `status`, the check's; `tested` and `reason`; `n_dose_groups_tested`; `tests_not_run` |

{api}`DoseResponseResult.fit` computes a cell type's fit again, with the values, weights and moderated statistics behind its tables. `adjusted_p_value` adjusts the p-values of a cell type and test by Benjamini-Hochberg, as limma's `topTable` does, and `adjusted_p_value_across_groups` adjusts each test over every cell type that ran it, one family per test, for a report of calls across cell types. The tests of each dose are adjusted within the cell type and dose only. For each dose against the vehicle with control over a gene's several doses, as Dunnett's test gives it, pass the cell type's moderated fit and its dose contrasts to {api}`multiple_contrast_test` ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts) shows how); to call the doses across genes, use {api}`DoseResponseResult.lowest_effective_doses` ([From which dose a gene responds](#lowest-effective-dose)).

A cell type compares the dose groups that keep a sample there and that the design can compare with the vehicle; the provenance lists them with their samples. A test whose contrasts a cell type's doses cannot estimate is not run there, such as a spline of 2 columns on the vehicle and two treated doses, and `tests_not_run` names it, with the reason in the record and in the error of `result.table`. When the dose groups are few, two tests can ask the same question: with the vehicle and two treated doses, a spline of one column and `dose_treated` pass through the three groups' means, so the spline's F-test is the dose-group test; with the vehicle on the axis and three doses, a spline of one column is the straight line. By default a cell type then runs the first of `spline`, `dose_groups` and `linear_trend` and lists the others as not run; a test named in `tests` runs whatever it repeats.

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
DoseResponseResult (limma_voom): 3 cell types, 2 tested; tests: spline (read
    first), dunnett, williams, mcp
  dose: ug/kg (administered) on a log10 axis, the vehicle in a group of its own;
        spline of 2 columns, knots at 0.548 ug/kg, boundary knots at 0.01 and 30
        ug/kg
  hepatocyte: 9 dose groups, 27 samples, 817 genes; calls at 5% FDR: spline 88,
              dunnett 69, williams 73, mcp 92
  stellate: 8 dose groups, 24 samples, 494 genes; calls at 5% FDR: spline 43,
            dunnett 24, williams 26, mcp 49
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

(lowest-effective-dose)=
## From which dose a gene responds

From which dose does each gene respond? ICH E4 asks that the lowest dose recommended for use itself show an effect [@ich1994], and toxicology reads the lowest observed effect level (LOEL) of each endpoint. Calling each dose against the vehicle in every gene would multiply the false calls by the doses and the genes, so {api}`DoseResponseResult.lowest_effective_doses` calls them in two stages, as stageR does [@vandenberge2017]:

1. The screen, a test of any response of the gene, the spline by default, calls the genes whose Benjamini-Hochberg adjusted p-value is at most `alpha`: R of the G genes it tested.
2. In each gene called, the doses' moderated t-tests against the vehicle, those of the table `each_dose`, are called by Holm's step-down procedure at the level R alpha / G that the screen leaves, with Shaffer's first step: the first multiplier is the gene's doses minus one, not the doses, since the screen has rejected that every dose equals the vehicle [@holm1979; @shaffer1986].

The lowest effective dose is the lowest dose called. The calls control the overall false discovery rate (OFDR) at `alpha`: the expected share of the genes called with at least one false call, the screen's or a dose's, proved under independence between genes [@heller2009] and expected wherever Benjamini-Hochberg controls the false discovery rate [@vandenberge2017]. They do not control the share of false calls among all gene-dose calls, which counts a gene once for each dose called. [Calling contrasts across genes](multiple_testing.md#stage-wise-testing) gives the formula.

```python
lowest = result.lowest_effective_doses("hepatocyte")
called = lowest.filter(lowest["responds"])
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

The table has one row per gene tested: `responds` says whether the screen calls the gene; `lowest_effective_dose` gives the lowest dose called, in the axis's unit, with that dose's `log2_fold_change` and `stage_wise_adjusted_p_value`; and `n_doses_called` counts the doses called. The lines above show the five genes of smallest stage-wise adjusted p-value. Of the 88 genes the spline calls in the hepatocytes, 69 have a dose called. A dose is called when its stage-wise adjusted p-value is at most `alpha`, as stageR counts, and the values hold only at the `alpha` given, since R depends on it: pass `alpha` to call at another level, rather than reading the values against another threshold. `screen` names another test of any response that the cell type ran, `"dose_groups"`, `"dunnett"`, `"williams"` or `"mcp"`; `"linear_trend"` and `"nonlinearity"` test a line or a departure from one, not any response.

An effective dose here is a dose whose change is detectable at the design, with no threshold on its size: the clinical minimum effective dose usually adds a clinically relevant change, as MCP-Mod's target dose does [@bornkamp2026]. It is a tested dose, not a potency between doses, and it moves with the replicates, the spacing of the doses and `alpha`; a gene can respond with no dose called, when its response spreads over several doses that each fall short. On simulated series, the overall false discovery proportion of these calls was 0.033 to 0.057 (standard errors 0.006 to 0.011), and the lowest effective dose lay a median of 8.9 to 14 times above the gene's true benchmark dose of a 1.1-fold change ([Validation](../validation.md#validation-dose-response-calibration)).

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
- **The tests.** Every test the design supports, of which `spline` is read first; in a cell type whose dose groups are few, one of two tests of the same hypothesis ([Which genes respond to dose](#dose-response-tests)). Dunnett's, the Williams-type and the MCP tests run only when named, two-sided ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts)).
- **Candidate shapes.** Five, spread over the design's doses by {api}`CandidateShapes.spread`: two Emax shapes, a sigmoid Emax shape of Hill slope 3, an exponential shape and a beta shape peaking mid-range.
- **The lowest effective dose.** The spline as screen, at `alpha=0.05`, each dose called by Holm's step-down procedure with Shaffer's first step at the level the screen leaves ([From which dose a gene responds](#lowest-effective-dose)).
- **The analysis.** {api}`limma_voom` with exact standard errors, its expression filter and TMM normalization, on the model of the dose groups.
- **The spline.** 2 columns when the axis holds 4 doses or more, otherwise 1, in the restricted cubic form, with knots at quantiles of the distinct doses of the design's samples.
- **FDR.** Benjamini-Hochberg within each cell type and test, and across the cell types that ran a test. With 2 replicates per dose, a cell type in which no gene responds has a call more often than 1 time in 20 ([Limits and differences](#dose-response-limits)).
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
- **Descriptive cell types.** {api}`dose_response` lists them with the check's reason and gives them no estimates; curves that describe their dose groups' means are not implemented yet.
- **Each dose against the vehicle.** The table `each_dose` adjusts each dose's tests within the cell type and dose, without control over the several doses of a gene, which {api}`multiple_contrast_test` on the cell type's fit gives dose by dose and `"dunnett"` for the gene as a whole ([Trends, doses against the vehicle and candidate shapes](#dose-response-contrasts)), or across genes, which {api}`DoseResponseResult.lowest_effective_doses` gives.
- **Degrees of freedom that are not integers.** The multiple contrast tests take a gene's moderated degrees of freedom as they are, where DoseFinding's `MCTtest` and multcomp's `glht` refuse non-integer ones ([Validation](#validation-dose-contrasts)).
- **Degrees of freedom under dream.** Under {api}`dream_voom` each contrast has its own degrees of freedom, and the multivariate t takes the smallest of the gene's doses', an approximation. It is conservative for Dunnett's test, whose contrasts are the doses. In a simulated design of 4 donors and 5 dose groups with 5 of its 20 samples left out, the Williams-type test's contrasts had more degrees of freedom than any dose, but the MCP step's had fewer in 19 to 174 of 240 genes, depending on the shape, down to 0.83 times the smallest dose's.
- **One set of contrasts per cell type.** The Williams-type and MCP contrasts follow a cell type's samples per dose group, as the spline does, not each gene's precision; each gene's covariance, voom's weights included, still sets the contrasts' standard errors and correlation.
- **Holm within each gene.** The per-dose calls of {api}`DoseResponseResult.lowest_effective_doses` leave out the correlation of a gene's doses through their shared vehicle, which the single-step max-t test uses. The gain it gives up is large with few degrees of freedom and small with many: at single-step p-values of 1e-6, Bonferroni's p-values are 22% larger for 3 doses and 65% for 8 at 5 degrees of freedom, and 1% and 3% at 40 ([Several contrasts of one gene](multiple_testing.md#correlated-tests)).
- **Responses with no dose called.** A gene that the screen calls can have no dose called, when its change spreads over several doses that each fall short; the lowest effective dose is a tested dose, not a potency between doses.
- **Non-monotone responses.** The Williams-type test assumes the response grows in dose order and misses many bells; the spline and the MCP step detect most of them, the MCP step partly through its beta shape: many bells turn back only at the top doses, where they look monotone.
- **The spline's weights.** The spline passes through the dose groups' means weighted by their samples, not by each gene's precision, so under voom's weights a dose group of few cells counts as much as its samples, though its mean is less precise.
- **Bands where the spline cannot follow.** A band holds the spline fitted to the dose groups' true means, which misses a response that jumps or turns more sharply than the spline can follow; read the band beside each dose's estimate ([Curves with a band](#dose-response-curves)).
- **Two replicates per dose.** In series where no gene responds, a cell type of 1,000 genes had a call at 5% FDR in 6.4% to 18.3% of the datasets with 2 replicates per dose, depending on the test and the number of doses, and in 4.5% to 7.5% with 3, the multiple contrast tests included, where Benjamini-Hochberg aims at 5% ([Validation](../validation.md#validation-dose-response-calibration)). With few residual degrees of freedom, 4 for the vehicle and three doses in duplicate, moderation pulls each gene's variance most of the way to a common prior, which understates the variance of genes that vary more than most between replicates. At 2 replicates per dose, read a few calls in a cell type as possibly due to chance.
- **limma's standard errors.** With `contrast_standard_errors="limma"`, the F-tests of the spline use limma's approximate standard errors, which depend on the spline's columns: the F-statistics of R's `ns()` columns and of the restricted cubic columns of the same spline differ by up to half ([Validation](#validation-dose-response-tests)). The exact default and {api}`limma_trend` do not depend on them. The multiple contrast tests refuse limma's standard errors, as {api}`multiple_contrast_test` does: they give no covariance of a gene's doses' estimates, and on a simulated series of 9 dose groups the Williams-type and MCP p-values moved by up to 100-fold with them.
- **Curves of dream fits and beyond the doses.** {api}`DoseResponseResult.curve` gives no curve for results of {api}`dream_voom`, whose tests run in full, and no value outside the doses a cell type compares.
- **What a simulated series leaves out.** In {api}`simulate_dose_series`, a gene's biological SD is the same at every dose, the constant variance that BMDS fits by default; genes are independent given their curves, so gene-set methods meet no correlation between genes; each cell type draws its responsive genes on its own, so few genes respond in several cell types; bells are symmetric, and no curve is hormetic; the counts hold no ambient RNA; and library sizes move with dose, by a median of 0.08 log2 at the top dose of default series, which TMM removes for the most part, leaving about 0.01 log2 on the flat genes at the top dose.

(dose-response-checks)=
## Checks and further reading

Dose axes, dose designs, dose tests and simulated dose series are compared with R, limma, scanpy, mpmath and published data, run as separate programs on the same inputs:

- {api}`DoseAxis`: positions, knots, columns, designs, F-tests and fitted curves against R 4.4.3's `ns()`, Hmisc's `rcspline.eval`, `model.matrix`, `lm` and `predict`, exact (1e-10); moderated F-tests of `limma_trend` against limma 3.62.1's, exact on `ns()`'s columns and close (1e-6) on Hmisc's, with every column counted, and equal in the two forms ([Validation](#validation-dose-axes)).
- {api}`check_dose_design`: cells, counts and the samples used per dose group against scanpy 1.12.4 and pandas; the residual degrees of freedom, estimable dose groups and groups confounded with a batch against R 4.4.3's `qr()` and limma 3.62.1's `nonEstimable`; statuses and ETAP's minimums against the same rules coded in R; and GSE208161, where every cluster is descriptive, all identical ([Validation](#validation-dose-design)).
- {api}`simulate_dose_series` and {api}`DoseTruth`: true curves, EC50s and benchmark doses against mpmath 1.3.0, exact (1e-10); the means and variances of cells and counts within 4 standard errors of their exact values; the same series at 1, 2 and 8 threads ([Validation](#validation-simulate-dose-series)).
- {api}`dose_response`: the samples, dose groups and genes of each cell type, its F-tests, slopes, each dose against the vehicle, adjusted p-values and curves with their bands against limma 3.62.1 and edgeR 4.4.0 run on R's own knots and spline columns, the exact default against each gene's weighted least squares in R, exact (1e-10); the spline's sums of squares against rms 8.2-1, exact; the dream route against variancePartition 1.36.2, close (1e-6); DRomics, dreamlet and muscat on the same profiles, as information ([Validation](#validation-dose-response-tests)); error rates under the null, power and the coverage of the bands on simulated series ([Validation](#validation-dose-response-calibration)).
- {api}`plot_dose_response`: the drawn curves, bands and points against limma-trend's spline in R, exact ([Validation](#validation-plotting)).
- {api}`CandidateShapes`, {api}`optimal_contrasts`, the three multiple contrast tests, {api}`stage_wise_adjusted_p_values` and {api}`DoseResponseResult.lowest_effective_doses`: shapes against DoseFinding 1.4-2's `Mods`, `getResp` and `guesst`; contrasts against `optContr` and `MCTtest`; the tests' statistics against limma 3.62.1 and variancePartition 1.36.2 fits in R with multcomp 1.4-32, exact (1e-10), close (1e-6) for dream; their p-values against mvtnorm 1.4-2 and scipy 1.18.1 within simulation error; stage-wise p-values and lowest effective doses against stageR 1.28.0 and a closed test enumerated in R, exact and identical ([Validation](#validation-dose-contrasts)); error rates and power on simulated series ([Validation](#validation-dose-response-calibration)).

Further reading:

- @desquilbet2010 explain dose-response analyses with restricted cubic splines, their knots and their tests of any response and of departure from a line.
- @ntp2018 sets out the National Toxicology Program's approach to genomic dose-response modelling, from its ANOVA gate to benchmark doses.
- @lun2017 showed that plate effects confounded with the condition break the control of false positives, and that summing the counts within each plate restores it.
- @crowell2020 benchmarked differential state methods on simulated and real multi-sample data, and found pseudobulk methods performing decently from about 100 cells per cell type and sample.
- @usepa2024 sets out the agency's standard methods for transcriptomic assessment products, with the design minimums of Sections 3.4.1.3, 3.4.2 and 3.4.5.2.
- @ema2014 qualified MCP-Mod for phase 2 dose finding, with its minimum numbers of doses; @bretz2005 introduced the combination of multiple comparisons and modelling behind it, and @pinheiro2014 its generalization to estimates and their covariance from any model; @duda2022 applied it gene by gene to expression data.
- @ich1994 sets out the dose-response information that supports drug registration, including the effect of the lowest recommended dose; @heller2009 and @vandenberge2017 describe the stage-wise testing behind the lowest effective doses.
- @nault2022 benchmarked tests of single-cell dose series simulated from the EPA's benchmark-dose equations; @crowell2020 and @song2024 describe the simulators of muscat and scDesign3; @crump1984 introduced the benchmark dose, and @kappenberg2023 reviews the design and analysis of 5,670 toxicological dose-response analyses.

[Gene-set tests](gene_set_tests.md) begins the part that turns the results for genes into results for pathways and regulators.
