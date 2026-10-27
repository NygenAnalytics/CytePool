---
description: Linear mixed models with random intercepts fitted to every gene by REML, Satterthwaite's and Kenward-Roger's degrees of freedom, F-tests of several contrasts, dream's precision weights and moderated tests for repeated measures designs, and variance partitioning.
---

# Mixed models, REML and dream

When a study has several samples per donor (two conditions, several tissues, a time
course), the samples of a donor are correlated. A fixed effect per donor removes that
correlation, but it also absorbs every variable that is constant within a donor, such as
disease status or sex, which then cannot be tested. A random intercept per donor models
the correlation instead and keeps donor-level variables testable. dream applies this to
gene expression [@hoffman2021], and dreamlet to the pseudobulk profiles of every cell
type [@hoffman2026a]. [Repeated measures and donors](../guides/repeated_measures.ipynb)
shows dream and variance partitioning on data.

## The model

Every gene is fitted with the same model (lme4's notation `(1 | donor)`; @bates2015):

$$
y = X\beta + Z_1 b_1 + \dots + Z_K b_K + e, \qquad
b_k \sim N(0, \sigma_k^2 I), \qquad e \sim N(0, \sigma^2 W^{-1}),
$$

where $X$ holds the fixed effects, $Z_k$ assigns each sample to a level of the $k$-th
grouping factor (one random intercept per level), and $W$ holds the gene's precision
weights. Grouping factors may be crossed, `(1 | donor) + (1 | batch)`, or nested,
`(1 | donor/sample)`. Random slopes are not implemented yet.

## REML

{api}`fit_mixed_model` estimates the variance components by restricted maximum
likelihood [@patterson1971], as lme4's `lmer` does, into a {api}`MixedModelFit`. With
$\theta_k = \sigma_k / \sigma$ and $V = W^{-1} + \sum_k \theta_k^2 Z_k Z_k'$, the fixed
effects and $\sigma^2$ are profiled out, and the criterion minimized is lme4's profiled
REML deviance

$$
\log\det V + \log\det(X' V^{-1} X) + (n - p)\left(1 + \log\frac{2\pi r^2}{n - p}\right),
$$

with $r^2 = (y - X\hat\beta)' V^{-1} (y - X\hat\beta)$. Newton's method minimizes it
with exact derivatives [@harville1977]; a variance component whose estimate is zero is
held exactly at zero, the boundary that lme4 reports as a singular fit. Random effects
are predicted by their best linear unbiased predictors [@henderson1975]. `method="ml"`
maximizes the likelihood instead.

## Satterthwaite's degrees of freedom

The t-statistic of a contrast $c$ of the fixed effects is $c'\hat\beta / \sqrt{c' C c}$,
with $C$ the covariance matrix of the estimates. Its distribution is approximated by a t
distribution with Satterthwaite's degrees of freedom [@satterthwaite1946;
@giesbrecht1985],

$$
\nu = \frac{2 v^2}{g' A g}, \qquad v = c' C c,
$$

where $g$ holds the derivatives of $v$ with respect to the variance parameters and $A$
is their asymptotic covariance matrix. {api}`fixed_effect_tests` gives them in
{api}`FixedEffectTests`, computed as lmerTest does [@kuznetsova2017], with exact
derivatives where lmerTest takes numerical ones. With every component at zero,
$\nu = n - p$, the degrees of freedom of the fixed-effects model.

## Kenward and Roger's method

With few samples the variance components are estimated imprecisely, and $C$, computed
as if they were known, understates the variance of the estimates. Kenward and Roger
[@kenward1997] adjust it: with $\Sigma = \sum_i \phi_i G_i$ the covariance of the
observations, linear in the variance parameters $\phi$,
$P_i = -X'\Sigma^{-1} G_i \Sigma^{-1} X$,
$Q_{ij} = X'\Sigma^{-1} G_i \Sigma^{-1} G_j \Sigma^{-1} X$ and $W$ the inverse of
the expected information of $\phi$ under REML,

$$
C_A = C + 2\,C \Big(\sum_{ij} W_{ij} \big(Q_{ij} - P_i C P_j\big)\Big) C .
$$

`fixed_effect_tests(fit, df_method="kenward_roger")` divides each estimate by its
standard error from $C_A$, with Satterthwaite's formula for the degrees of freedom on
the expected information, as pbkrtest computes them [@halekoh2014] and lmerTest's
`ddf = "Kenward-Roger"` reports them. In balanced designs the adjustment vanishes and
the tests are those of the analysis of variance. A variance component estimated at
zero takes part, which lowers the degrees of freedom below Satterthwaite's.

With precision weights, pbkrtest builds $\Sigma$ without lme4's prior weights while
it takes $C$ from the weighted fit. dream divides each gene's weights by their mean,
so its tests treat every observation as having its gene's average weight.
`kenward_roger_weights="average"`, the default, computes these tests; `"precision"`
uses each observation's own weight, which is Kenward and Roger's method for the
weighted model. With voom's weights the first can make standard errors smaller than
the unadjusted ones, and the second never does; the [Validation](../validation.md)
page compares them.

## Several contrasts at once

{api}`fixed_effect_f_test` tests that several contrasts are all zero, such as every
difference among three time points, in a {api}`FixedEffectFTest`. With Satterthwaite's
method, as lmerTest's `contestMD`, the contrasts are replaced by the eigenvectors of
their covariance matrix, whose estimates are uncorrelated; $F$ is the mean of their
squared t-statistics, and its denominator degrees of freedom $2E/(E-r)$, with
$E = \sum_m \nu_m / (\nu_m - 2)$ over their degrees of freedom, match the mean of
$F$ [@fai1996; @kuznetsova2017]. With Kenward and Roger's method, as pbkrtest's
`KRmodcomp`, the Wald statistic with $C_A$ is scaled and referred to an $F$
distribution whose first two moments it matches [@kenward1997]. The numerator degrees
of freedom $r$ are the number of linearly independent contrasts.

## dream

dream is limma-voom with the mixed model in place of the linear model:

1. {api}`voom_with_dream_weights`: voom's precision weights, from the mean-variance
   trend of the mixed model of every gene, fitted by maximum likelihood, read at the
   fitted values with the predicted random effects.
2. {api}`fit_mixed_model` by REML with those weights, and {api}`fixed_effect_tests`
   with dream's degrees of freedom: by default Kenward and Roger's for fewer than 20
   samples, as pbkrtest computes them with every observation at the average weight,
   and Satterthwaite's otherwise.
3. {api}`moderated_fixed_effect_tests`: empirical Bayes moderation of each gene's
   residual variance on the approximate residual degrees of freedom of the mixed model,
   $\operatorname{tr}((I - H)'(I - H))$ for its hat matrix $H$; the moderated
   t-statistic has the degrees of freedom of its test plus the prior's, and the result
   ({api}`ModeratedFixedEffectTests`) adds a z-statistic with the same p-value, so that
   tests on different degrees of freedom compare. Its moderated F-statistic,
   $t' R^{-1} t / r$ for the moderated t-statistics and the correlation $R$ of their
   estimates, tests several contrasts at once, as `topTable` does.

{api}`dream_voom` runs these steps on every cell type of a cube, with
`df_method="adaptive"` choosing the degrees of freedom in each cell type from the
samples left in it, and `f_tests` naming sets of contrasts to test at once. As
variancePartition does, it moderates the residual variance of an observation of each
gene's average weight, which keeps the mean-variance trend that voom's weights remove;
`trend=True` lets the prior follow it.

dream switches from Kenward and Roger's method to Satterthwaite's at 20 samples
(its help page says "<= 20"; with 20 it uses Satterthwaite's), and for a gene whose
adjusted covariance matrix is ill conditioned, which depends on the units of
continuous covariates. Its `topTable` computes the p-values of an F-test of every gene
on the degrees of freedom of the first gene; here each gene has its own, those of $r$
uncorrelated moderated t-statistics combined as lmerTest combines those of its
F-tests, so that an F-test of one contrast is its t-test. The
[Validation](../validation.md) page reports both differences and their sizes.

## Variance partitioning

Which variables drive the differences between samples, and which must be adjusted for?
Variance partitioning [@hoffman2016] fits every gene with one mixed model that holds
every variable of the design, the categorical ones as random intercepts (donor,
condition, batch) and the continuous ones as fixed effects (age), and reports the
fraction of the gene's variance that each variable explains:

$$
\frac{\hat\sigma^2_v}{\sum_k \hat\sigma^2_k + \sum_j \hat\sigma^2_{\beta_j} +
\hat\sigma^2},
$$

where a random effect $k$ contributes its variance component $\hat\sigma^2_k$, a
fixed-effect term $j$ the variance of its fitted contribution over the samples,
$\hat\sigma^2_{\beta_j} = \operatorname{var}(X_j \hat\beta_j)$, and the residual its
variance $\hat\sigma^2$. The fractions of a gene sum to one; without fixed effects,
the fraction of a random effect is the intra-class correlation of two samples that
share its level and no other.

{api}`variance_fractions` computes them into a {api}`VarianceFractions`, as
variancePartition's `fitExtractVarPartModel` does:

- The variances are estimated by maximum likelihood, variancePartition's default: REML
  estimates the variances after removing the fixed effects, which biases the fractions
  of the fixed effects. `method="reml"` fits by REML.
- With precision weights, such as voom's, each gene's weights are divided by their
  mean, so that the residual variance is that of an observation of average weight, and
  the variances of the fitted fixed effects are weighted.
- With random effects, the fixed-effect terms share the variance of all the fitted
  fixed effects in proportion to the variances of their own contributions (found by
  probing variancePartition). A design without random effects is fitted by least
  squares, and each of its terms, factors included, has the variance of its own
  contribution.
- A variance component estimated at zero, a singular fit in lme4, gives a fraction of
  exactly zero.

{api}`variance_partitioning` partitions the variance of every cell type of a cube
after the steps of {api}`dream_voom` up to voom's precision weights, which come from
the mean-variance trend of the mixed model, and records the samples each fit used in
a {api}`VariancePartitioningResult`; {api}`VariancePartitioningResult.table` gives the
fractions of one cell type. {api}`plot_variance_fractions` draws variancePartition's
violins of the fractions over the genes [@hintze1998], with Silverman's bandwidth
[@silverman1986], and {api}`plot_feature_variance_fractions` its bars for chosen genes.
The residual fraction mixes biological variation, the precision of the measurements
and technical variation that the design does not name, so its size alone does not say
whether variables are missing from the design.

## Not yet implemented

Random slopes are planned; they would also partition the variance within subsets of
the samples, such as the variance between donors within each tissue. A gene that the
fixed and random effects reproduce exactly has no REML estimate; it is left unfitted
rather than given a variance of zero.
