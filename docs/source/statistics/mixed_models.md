---
description: Linear mixed models with random intercepts fitted to every gene by REML, Satterthwaite's degrees of freedom, and dream's precision weights and moderated tests for repeated measures designs.
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

## Not yet implemented

Kenward-Roger's degrees of freedom (dream's default for 20 samples or fewer), F-tests of
several contrasts at once, random slopes and variance partitioning [@hoffman2016] are
planned. A gene that the fixed and random effects reproduce exactly has no REML
estimate; it is left unfitted rather than given a variance of zero.
