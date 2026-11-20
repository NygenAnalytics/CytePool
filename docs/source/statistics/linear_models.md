---
description: Which comparison a design asks of the samples, how one linear model fits every gene, which contrasts the samples can estimate, how splines of dose enter the design, and simulated dose series.
---

# Designs, linear models and contrasts

[Pseudobulk profiles and cubes](cubes.md) gave every sample a profile in each cell type. Which genes change is then a question about those samples: which of them are compared, and what else differs between them. A {api}`Design` states it: the sample table, the fixed effects as an R formula, and named contrasts; {api}`fit_linear_model` fits it to every gene at once, as limma's `lmFit` does, and {api}`contrasts_fit` turns each gene's fit into the contrasts. {api}`limma_voom` takes these steps in every cell type, and [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) builds a paired design on published data. Splines of dose enter a design through a dose axis, and a simulated dose series with known curves shows how well a dose method recovers them.

(designs-and-formulas)=
## The design

A design starts from the sample table: one row per sample, with a `sample_id` and the variables of the question, such as the condition and the donor. It is matched to the samples of a cube by `sample_id`, never by position, so it may name only some of the cube's samples, and in any order.

The fixed effects are an R formula, in the notation of @wilkinson1973 as R reads it
[@chambers1992]: `~ condition + batch` adds terms, `a:b` is an interaction, `a * b`
expands to `a + b + a:b`, `a / b` nests `b` within `a`, `(a + b + c)^2` holds the terms up to two-way interactions, and `~ 0 + group` has no intercept. {api}`model_matrix` builds the same columns as R's `model.matrix`: a numeric column is a covariate, and a text or boolean column is a factor, coded by treatment contrasts (an indicator for each level but the first), or by an indicator for every level where R codes it so, as in `~ 0 + group`. Columns are named as R names them: the intercept `(Intercept)` ({api}`INTERCEPT`), a level after its factor (`conditionstim`), interactions joined by `:`, and names that are not syntactic in R between backticks ({api}`quote_name`). {api}`parse_formula` expands a formula into R's terms ({api}`FormulaTerms`) and {api}`model_matrix` returns the matrix with its column names ({api}`ModelMatrix`); their entries in the API reference give the grammar and the coding in full.

In a design with an intercept, the first level of a factor is its reference, and every coefficient of the factor is a difference from it. Without `levels`, the levels of a pandas categorical follow its categories, and those of a text column are sorted by code point; pass `levels={"condition": ["ctrl", "stim"]}` whenever the reference matters, which is almost always. Random effects, written in lme4's notation as in `random="(1 | donor)"` [@bates2015] and read by {api}`parse_random_effects` into {api}`RandomEffect` terms, are for mixed models ([Mixed models and dream](mixed_models.md)); the models of this chapter have fixed effects.

(one-model-per-gene)=
## One model for every gene

Every gene $g$ of a samples x genes matrix of log-expression values is fitted with the same design matrix $X$ (limma's `lmFit`; @smyth2004):

$$
E(y_g) = X a_g, \qquad \operatorname{var}(y_g) = \sigma_g^2\, W_g^{-1/2} R\, W_g^{-1/2},
$$

where $W_g$ holds the gene's weights, such as voom's precision weights or sample weights, and $R$ is the identity, or the correlation matrix of a known correlation between the samples of a block, such as those of one donor [@smyth2005]. {api}`fit_linear_model` returns a {api}`LinearModelFit` with, for every gene, the coefficients $\hat a_g$, the residual variance $s_g^2$ on $d_g$ residual degrees of freedom, and the unscaled standard errors $u_{gk}$, so that the standard error of coefficient $k$ is $u_{gk} s_g$. When no value is missing and the weights, if any, are positive sample weights, one factorization of the design serves every gene. A gene with missing values, or with weights of its own, is fitted to its observed samples by a QR decomposition of its own [@golub2013]: missing values and zero weights lower its $d_g$, and a coefficient that its remaining samples cannot estimate is NaN.

(contrasts)=
## Contrasts

A contrast is a combination $c$ of the coefficients that answers one question, such as the difference between two groups. {api}`make_contrasts` reads named expressions, as in `"groupB - groupA"` and `"(groupB + groupC)/2 - groupA"`, written in the design's column names, as limma's `makeContrasts` does [@ritchie2015], into {api}`Contrasts`, a mapping of contrast names to weights; {api}`Contrasts.select` takes some of them by name, and {api}`Design.with_contrasts` gives the design other contrasts. {api}`contrasts_fit` re-expresses each gene's fit: the estimate is $c' \hat a_g$, with the unscaled variance $c' V_g c$, where $V_g$ is the gene's unscaled covariance matrix. Weights that name their coefficients, such as a {api}`Contrasts` or a pandas Series, are matched to them by name; a plain vector or matrix is read in the order of the coefficients.

When one factorization serves every gene, $V_g$ is the same for every gene. Otherwise each gene has its own, and limma's `contrasts.fit` approximates the standard error of a contrast from the gene's own standard errors and the correlations of the coefficients in the design without weights. The approximation is exact for contrasts of uncorrelated coefficients, such as the group means of a one-way design, and otherwise depends on how the design is written. In `~ condition + age` with three conditions, stimulated against control is one coefficient when control is the reference level and the difference of two coefficients when another level is, and its approximate standard error changes with the choice. CytePool computes the standard errors exactly by default, from each gene's own $V_g$, so that every parametrization of a design gives the same tests; `method="limma"` reproduces limma's approximation, which can be far from the exact value ([Validation](#validation-linear-models) compares them).

(paired-design)=
## A paired design

When every donor gives a sample in each condition, the design `~ donor + condition` gives each donor a coefficient of its own, so that the conditions are compared within donors and the differences between donors do not count as noise; `conditionstim` then estimates the change that the stimulation makes within a donor. A variable that is constant within each donor, such as sex, age or disease, cannot be tested in this design: its column is a combination of the donor columns, and {api}`Design.check` reports it as aliased and confounded with the donor. A random effect for the donor tests such a variable ([Mixed models and dream](mixed_models.md)), and so does a comparison of the profiles pooled by donor ({api}`Cube.collapse_samples`), when every donor was sampled alike.

(spline-bases)=
## Spline columns

How does expression change along a dose, a time or another number, when the shape is not known in advance? A natural cubic spline answers with a few columns of the design: cubic between its knots, joined smoothly at them, and straight beyond its two boundary knots. A linear model fits the columns' coefficients like any others, so voom's weights, moderation and F-tests ([F-tests of several contrasts](limma_voom.md#limma-f-tests)) apply unchanged. {api}`NaturalSplineBasis` builds the columns from the values that place its knots, keeps the knots, and records itself as plain values with {api}`NaturalSplineBasis.to_dict`. {api}`NaturalSplineBasis.matrix` gives the columns at other values of the same kind, such as a grid on which to draw a fitted curve. For a spline of dose those values are positions on the dose axis, such as log10 doses, and {api}`DoseAxis.design_columns` gives the columns at doses ([Dose axes and the vehicle](#dose-axes)).

The basis comes in two forms that span the same curves for the same knots, so their fits and the F-test of all their columns agree [@durrleman1989; @harrell2015; @perperoglou2019]. `form="b_spline"` gives the columns of R's `ns()` up to a change of basis, and {api}`natural_spline_basis` gives them at x. `form="restricted_cubic"` gives Harrell's restricted cubic spline with its linear part first: the first column is the value minus the lower boundary knot, so the F-test of the other columns tests departure from a straight line [@desquilbet2010]. With the first column, the other columns span the curves of Hmisc's `rcspline.eval` and rms's `rcs` [@harrell2026]; they are built from the B-splines, so that they stay well conditioned when knots crowd, where rms's truncated cubes become nearly collinear ([Limits and differences](#linear-models-limits)). Without an intercept, every column of either form is 0 at the lower boundary knot.

Default knots differ between tools, so a curve cannot be reproduced without its knots: {api}`NaturalSplineBasis` keeps them, and its repr prints them. A rule places them on the values, or they are given:

| Rule | Where the knots go | The same knots in R |
| --- | --- | --- |
| `"quantiles"` | K interior knots at the quantiles of the values with probabilities k / (K + 1), K being `df - 1` without an intercept and `df - 2` with one; the boundary knots at the range of the values | `attr(ns(x, df), "knots")` |
| `"levels"` | The same rule on the distinct values, each counted once, so that a dose with more replicates pulls no knot towards it; these knots never coincide while two or more distinct values lie within the boundary knots | `attr(ns(unique(x), df), "knots")` |
| `"harrell"` | All `df + 1` knots without an intercept and `df` with one, boundary knots included, at percentiles from 10% to 90% for 3 knots, 5% to 95% for 4 to 6, 2.5% to 97.5% for more, equally spaced, with the 5th smallest and largest values as the outer knots below 100 values | `rcspline.eval(x, nk = df + 1, knots.only = TRUE)`, or `rcs(x, df + 1)` |
| Given values | Where given, strictly inside the boundary knots | `ns(x, knots = k, Boundary.knots = b)` |

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

(simulated-dose-series)=
## Simulated dose series with known curves

How well does a dose method find the genes that respond, the shape of each response and the dose at which it starts, at a given design? Only data whose truth is known can say, and the one single-cell dose benchmark compared tests on cells, without curves [@nault2022]. {api}`simulate_dose_series` draws a cube of pseudobulk counts from known curves of each gene's expression along dose, and {api}`DoseTruth` holds the truth. Here the spline F-test of the previous section runs on a simulated series, and the truth says which genes it should call:

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

The benchmark dose of a 1.1-fold change of Hill, exponential, Fréchet and step curves is log-uniform from an eighth of the log10 span of the positive doses below the lowest dose to an eighth above the top dose, so that a tenth lies below the doses and a tenth above. The EC50 follows from it, so about a quarter of Hill and exponential EC50s lie above the top dose, where the doses never reach half the maximal change, and over a third at 4 dose groups. Slopes are log-uniform from 0.5 to 4, and from 1 to 4 for power curves, which reach their maximal change at the top dose, as linear curves do. Bells peak within the doses, with an SD along log10 dose of 0.75 to 1.5 times the gap between the doses on either side of the peak, so that 98% to 99% reach half their peak at two or more doses, at six designs of 3 to 9 dose groups, evenly spaced or not. Their benchmark dose, the lower crossing, then lies below the lowest dose more often in sparse designs: in 31% of bells at nine dose groups (the vehicle and eight half-log doses) and 94% at four (the vehicle and three doses a decade apart) ([Validation](../validation.md#validation-simulate-dose-series)).

`expression_sd` is the SD of true expression between vehicle replicates, from biological and donor variation, before counting, and every truth is a change in true expression. Counting widens the SD of observed log-CPM values most for genes with few counts: at 100 cells per profile, 3.6 times for genes with 1 to 5 expected counts at the vehicle and 1.2 times above 100. The prior count of log-CPM values compresses the observed changes of the same genes: at the top dose and 100 cells, to 0.40 of the true change below 1 expected count, 0.80 from 1 to 2 and 0.95 from 2 to 5, falls more than rises, and to 0.99 or more from 5 counts on ([Validation](../validation.md#validation-simulate-dose-series)). For SDs on the observed scale, simulate the same seed with many vehicle replicates, such as `replicates=[2000, 3, 3, 3, 3, 3]`, which keeps the truth, and pass a multiple of each gene's observed SD to {api}`DoseTruth.benchmark_doses`. That pairs an observed SD with the true curve, so compare such benchmark doses only for genes with at least 5 expected counts at the vehicle and the top dose, or draw the observed curve, the mean log-CPM at each dose, from a series with many replicates of every dose.

The random draws come in independent streams, so a cell type's curves and expression stay the same when `replicates`, `cells`, `composition`, `biological_sd` or `donor_sd` change, or another cell type is added: designs compare on the same genes. Doses multiplied by a factor give EC50s, peak doses and benchmark doses multiplied by it, and the same counts.

(estimable-contrasts)=
## Designs that cannot answer the question

A contrast is estimable when the samples determine it: when it is a combination of the rows of $X$, or equivalently orthogonal to the null space of $X$ [@searle1971]. In a design of full rank every contrast is. A design loses rank when a column is a combination of others, decided as R's `qr()` decides it, which calls a column aliased when its part orthogonal to the columns before it is at most $10^{-7}$ of its length. That happens when two factors group the samples alike, when a variable is constant within the levels of another, or, in one cell type, when every sample of a donor was left out.

When a cell type's design loses rank, {api}`limma_voom` fits it on the columns kept by R's `qr()`, which are the ones limma's `lmFit` keeps, and tests each contrast that the cell type's samples can estimate. The estimate and its exact standard error are then those of any full-rank parametrization of the same model, even for a contrast that weighs a column the fit left out. A contrast that the samples cannot estimate is not computed, and the result names the relations it violates. limma does otherwise on both counts. Its `contrasts.fit` stops with an error for a contrast that weighs a column it left out, and so for every contrast passed with it, which in limma calls for a full-rank parametrization; and it computes a contrast that is not estimable from the columns it kept, whose meaning has then changed. With every sample of the reference donor D1 left out of `~ donor + condition`, limma reports for `donorD2`, donor D2 against D1, the difference between donor D2 and the last donor ([Validation](#validation-differential-expression)). A cell type left without residual degrees of freedom, or without a contrast it can estimate, is not fitted, and the result says why ([What runs in each cell type](limma_voom.md#limma-voom-steps)).

(design-checks)=
## Design checks

Most problems of a design can be found before anything is fitted. {api}`Design.check` returns a {api}`DesignReport` of {api}`DesignIssue` entries, whose messages say what to change, and {api}`DesignReport.kinds` lists their kinds. An error means that something the analysis needs cannot be estimated, as when columns are aliased, factors are confounded, a level is empty, no residual degrees of freedom are left, or the variance of a random effect cannot be estimated. A warning marks a design that fits but may not mean what was intended: a level with one sample; a sample fitted exactly, or nearly so, by the model, with a leverage (its diagonal element of the hat matrix [@hoaglin1978]) of 1 or above 0.99; a numeric column that looks like codes of batches or ids of donors, which would enter the model as one slope; or labels that look like missing values, or that differ in nothing but case or spaces and so split one group in two. [The designs page](/api/designs.html) of the API reference lists every check.

(linear-models-defaults)=
## Defaults and choices

- **The reference level.** The first level of each factor: in the order that `levels` gives, or else that of a pandas categorical's categories, or else by code point ([The design](#designs-and-formulas)).
- **Group means or differences.** `~ 0 + group` estimates the mean of each group, and its contrasts compare them, as `groupB - groupA` does; `~ group` estimates the reference group and the differences from it. With exact standard errors, both give the same tests of the same comparison ([Contrasts](#contrasts)).
- **Pairing or a random effect.** The paired design suits questions within donors; a random effect suits variables of the donors, and donors with unequal numbers of samples ([Mixed models and dream](mixed_models.md)).
- **Weights.** Sample weights, one per sample, or observation weights, one per sample and gene, such as voom's; a weight of zero leaves an observation out of the fit.
- **A block correlation.** `block` and `correlation` fit a known correlation between the samples of each block; with a correlation of at least zero, that is a random intercept whose correlation is given rather than estimated.
- **Contrast standard errors.** Exact by default; `method="limma"` reproduces limma's approximation ([Contrasts](#contrasts)).
- **The spline form.** The restricted cubic form in {api}`DoseAxis.spline_basis`, whose columns after the first test departure from a line. The B-spline form, R's, is the default in {api}`NaturalSplineBasis` and {api}`natural_spline_basis` ([Spline columns](#spline-bases)).
- **The knots.** At quantiles of the distinct doses in {api}`DoseAxis.spline_basis`, each dose counted once, so that the knots depend on the doses given and not on how many samples of each survived; R's quantile rule in {api}`NaturalSplineBasis` ([Spline columns](#spline-bases)).
- **The vehicle.** In a group of its own. On the axis, at 0 on a linear axis, or on a log10 axis only at an offset that is given and recorded ([Dose axes and the vehicle](#dose-axes)).

(linear-models-limits)=
## Limits and differences

- **Formulas that R reads otherwise.** Three forms whose R reading is seldom what was meant are refused: a `-` that removes nothing, as in `~ a - b`; `^` of a single term, as in `~ x^2`, which R reads as `~ x`; and a fractional power, which R truncates. Function calls such as `log(age)`, `I(age^2)` or `factor(w)` are refused too: add a column of the computed values to the sample table instead.
- **Level order.** Text levels are sorted by code point in every locale. R sorts them by the collation of its session's locale, which gives the same order only with `LC_COLLATE=C`.
- **Contrasts that are not linear.** {api}`make_contrasts` accepts linear combinations of columns only. limma's `makeContrasts` evaluates any expression, and turns `groupB - 1`, products, ratios and powers into weights without meaning.
- **Contrasts by name.** limma's `contrasts.fit` reads the rows of the weights in their order, and merely warns when their names do not match the coefficients' ([Contrasts](#contrasts)).
- **Coefficients the data do not determine.** A coefficient that a gene's remaining samples cannot estimate is NaN ([One model for every gene](#one-model-per-gene)). limma instead drops whichever collinear column comes later and reports the rest, which may include coefficients that the data do not determine.
- **Rank and random effects.** {api}`fit_linear_model` needs a design of full column rank, and refuses random effects, which it would drop, except one random intercept that `block` and `correlation` give; the message says what fits the design. limma's `duplicateCorrelation`, which estimates the correlation of the blocks, is not implemented.
- **Memory.** A fit with observation weights or missing values keeps a covariance matrix for every gene ({api}`fit_linear_model` gives its size).
- **Ordered factors.** An ordered pandas categorical is coded as an unordered factor, by treatment contrasts, where R would use polynomial contrasts.
- **Splines in formulas.** Formulas refuse `ns()`, `bs()` and `poly()`; add the columns of {api}`DoseAxis.design_columns` or {api}`NaturalSplineBasis.matrix` to the sample table instead.
- **The restricted cubic columns.** Hmisc's `rcspline.eval` and rms's `rcs` give the value itself and Harrell's truncated cubes; CytePool gives the value minus the lower boundary knot, so that every column is 0 there, and columns built from the B-splines that span the same curves. Fits, curves, the coefficient of `dose_treated` and the F-tests of all the columns and of the columns after the first are rms's; the other coefficients differ. With Hmisc's first column, the intercept, or `dose_treated` with the vehicle in its own group, also differs, by the first coefficient times the lower boundary knot.
- **Crowded knots.** Truncated cubes become nearly collinear when knots crowd, as on a linear axis over doses from 0.01 to 30, where designs with Hmisc's columns reach condition numbers of 1e11 and R's `qr()` drops columns as aliased. Where the correlation of their coefficients has an eigenvalue below 1e-8 of the largest, the F-tests of limma, and of CytePool, which follows limma's rule, count fewer degrees of freedom than columns: limma's did in 13 cases of the validation, on the log10 axis too. CytePool's restricted cubic columns stayed below a condition number of 1.3e3 on every axis, and the two forms gave the same moderated F-tests ([Validation](#validation-dose-axes)).
- **Harrell's knots.** Where Hmisc places fewer knots than asked, drops a repeated knot, or switches to another method because 5% of the values or more tie at the smallest or largest value and none between them ties as often, `"harrell"` raises an error instead. {api}`DoseAxis.spline_basis` takes `"levels"`, `"quantiles"` or given knots: on a ladder with fewer than 5 samples per dose, Harrell's outer knots fall inside the lowest and highest doses, and `dose_treated` would no longer compare the lowest dose with the vehicle.
- **Repeated knots.** R's `ns()` and {api}`natural_spline_basis` accept repeated knots, which lower the curve's smoothness there, as R's quantile rule gives on tied values; the restricted cubic form refuses them, and {api}`DoseAxis.spline_basis` refuses knots that coincide in either form, naming the rule and the dose.
- **Doses that differ by rounding.** {api}`DoseAxis.spline_basis` refuses two doses that differ by less than 1e-8 of the dose, as `0.1 * 3` and `0.3` do, which a conversion of units can leave as two doses where there is one: round the dose column first.
- **What a simulated series leaves out.** In {api}`simulate_dose_series`, a gene's biological SD is the same at every dose, the constant variance that BMDS fits by default; genes are independent given their curves, so gene-set methods meet no correlation between genes; each cell type draws its responsive genes on its own, so few genes respond in several cell types; bells are symmetric, and no curve is hormetic; the counts hold no ambient RNA; and library sizes move with dose, by a median of 0.08 log2 at the top dose of default series, which TMM removes for the most part, leaving about 0.01 log2 on the flat genes at the top dose.

(linear-models-checks)=
## Checks and further reading

Designs and fits are compared with R and limma, run as separate programs on the same data:

- Model matrices, R's terms, the rank and aliased columns of designs, leverages and the estimability of contrasts: R 4.4.3, exact (identical, and 1e-10; 1e-8 absolute for the leverages of badly scaled designs); named contrasts: limma 3.62.1; random-effect terms: lme4 2.0-6 ([Validation](#validation-design)).
- {api}`fit_linear_model` and {api}`contrasts_fit`: `lmFit` and `contrasts.fit` of limma 3.62.1, exact (1e-10) ([Validation](#validation-linear-models)).
- {api}`NaturalSplineBasis` and {api}`natural_spline_basis`: knots identical to those of R 4.4.3's `ns()` and Hmisc 5.3-1's `rcspline.eval`; bases at new points against `predict()`, the restricted cubic columns' first column and the curves they span against `rcspline.eval`, and F-tests against rms 8.2-1, exact (1e-10) ([Validation](#validation-smoothing)).
- {api}`DoseAxis`: positions, knots, columns, designs, F-tests and fitted curves against R 4.4.3's `ns()`, Hmisc's `rcspline.eval`, `model.matrix`, `lm` and `predict`, exact (1e-10); moderated F-tests of `limma_trend` against limma 3.62.1's, exact on `ns()`'s columns and close (1e-6) on Hmisc's, with every column counted, and equal in the two forms ([Validation](#validation-dose-axes)).
- {api}`simulate_dose_series` and {api}`DoseTruth`: true curves, EC50s and benchmark doses against mpmath 1.3.0, exact (1e-10); the means and variances of cells and counts within 4 standard errors of their exact values; the same series at 1, 2 and 8 threads ([Validation](#validation-simulate-dose-series)).

Further reading:

- @wilkinson1973 introduced the symbolic description of factorial models from which the formulas of S and R grew, and @chambers1992 described the formulas and the coding of factors of S, which R follows.
- @smyth2004 set out limma's approach to experiments with few samples: one linear model fitted to every gene, whose contrasts are tested by moderated statistics.
- @searle1971 gave the theory of linear models of less than full rank, and of the functions of their coefficients that the data can estimate.
- @hoaglin1978 described the hat matrix, whose diagonal gives the leverage of each sample.
- @durrleman1989 introduced restricted cubic splines to medical regression, @harrell2015 sets out their use with knots at percentiles, and @desquilbet2010 shows how to test overall and non-linear dose-response associations with them; @perperoglou2019 reviews the spline functions of R, `ns()` among them.
- @nault2022 benchmarked tests of single-cell dose series simulated from the EPA's benchmark-dose equations; @crowell2020 and @song2024 describe the simulators of muscat and scDesign3; @crump1984 introduced the benchmark dose, and @kappenberg2023 reviews the design and analysis of 5,670 toxicological dose-response analyses.

[Empirical Bayes moderation](empirical_bayes.md) makes the tests of these models hold with few samples.
