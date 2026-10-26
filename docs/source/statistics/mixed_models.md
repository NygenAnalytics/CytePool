---
description: Linear mixed models with random intercepts fitted to every gene by REML, Satterthwaite's degrees of freedom, dream's precision weights and moderated tests for repeated measures designs, and variance partitioning.
---

# Mixed models, REML and dream

When a study has several samples per donor (two conditions, several tissues, a time
course), the samples of a donor are correlated. A fixed effect per donor removes that
correlation, but it also absorbs every variable that is constant within a donor, such as
disease status or sex, which then cannot be tested. A random intercept per donor models
the correlation instead and keeps donor-level variables testable. dream applies this to
gene expression [@hoffman2021], and dreamlet to the pseudobulk profiles of every cell
type [@hoffman2026a].

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

## dream

dream is limma-voom with the mixed model in place of the linear model:

1. {api}`voom_with_dream_weights`: voom's precision weights, from the mean-variance
   trend of the mixed model of every gene, fitted by maximum likelihood, read at the
   fitted values with the predicted random effects.
2. {api}`fit_mixed_model` by REML with those weights, and {api}`fixed_effect_tests`.
3. {api}`moderated_fixed_effect_tests`: empirical Bayes moderation of each gene's
   residual variance on the approximate residual degrees of freedom of the mixed model,
   $\operatorname{tr}((I - H)'(I - H))$ for its hat matrix $H$; the moderated
   t-statistic has Satterthwaite's degrees of freedom plus the prior's, and the result
   ({api}`ModeratedFixedEffectTests`) adds a z-statistic with the same p-value, so that
   tests on different degrees of freedom compare.

{api}`dream_voom` runs these steps on every cell type of a cube. As variancePartition
does, it moderates the residual variance of an observation of each gene's average
weight, which keeps the mean-variance trend that voom's weights remove; `trend=True`
lets the prior follow it.

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
[@silverman1986], and {api}`plot_gene_variance_fractions` its bars for chosen genes.
The residual fraction mixes biological variation, the precision of the measurements
and technical variation that the design does not name, so its size alone does not say
whether variables are missing from the design.

## Not yet implemented

Kenward-Roger's degrees of freedom (dream's default for 20 samples or fewer), F-tests of
several contrasts at once and random slopes are planned; random slopes would also
partition the variance within subsets of the samples, such as the variance between
donors within each tissue. A gene that the fixed and random effects reproduce exactly
has no REML estimate; it is left unfitted rather than given a variance of zero.
