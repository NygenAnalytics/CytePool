---
description: Which comparison a design asks of the samples, how one linear model is fitted to every gene, and which contrasts of its coefficients the samples can estimate.
---

# Designs, linear models and contrasts

[Pseudobulk profiles and cubes](cubes.md) gave every sample a profile in each cell type. Which genes change is then a question about those samples: which of them are compared, and what else differs between them. A {api}`Design` states it: the sample table, the fixed effects as an R formula, and named contrasts; {api}`fit_linear_model` fits it to every gene at once, as limma's `lmFit` does, and {api}`contrasts_fit` turns each gene's fit into the contrasts. {api}`limma_voom` takes these steps in every cell type, and [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) builds a paired design on published data.

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

(linear-models-checks)=
## Checks and further reading

Designs and fits are compared with R and limma, run as separate programs on the same data:

- Model matrices, R's terms, the rank and aliased columns of designs, leverages and the estimability of contrasts: R 4.4.3, exact (identical, and 1e-10; 1e-8 absolute for the leverages of badly scaled designs); named contrasts: limma 3.62.1; random-effect terms: lme4 2.0.6 ([Validation](#validation-design)).
- {api}`fit_linear_model` and {api}`contrasts_fit`: `lmFit` and `contrasts.fit` of limma 3.62.1, exact (1e-10) ([Validation](#validation-linear-models)).

Further reading:

- @wilkinson1973 introduced the symbolic description of factorial models from which the formulas of S and R grew, and @chambers1992 described the formulas and the coding of factors of S, which R follows.
- @smyth2004 set out limma's approach to experiments with few samples: one linear model fitted to every gene, whose contrasts are tested by moderated statistics.
- @searle1971 gave the theory of linear models of less than full rank, and of the functions of their coefficients that the data can estimate.
- @hoaglin1978 described the hat matrix, whose diagonal gives the leverage of each sample.

[Empirical Bayes moderation](empirical_bayes.md) makes the tests of these models hold with few samples.
