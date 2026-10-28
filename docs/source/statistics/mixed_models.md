---
description: Which genes change when donors give several samples each, tested by dream's mixed models with a random intercept per donor, and which variables drive the variation, by variance partitioning.
---

# Mixed models and dream

[edgeR and DESeq2](count_models.md), like limma-voom, take the samples as independent. When
donors give several samples each, under two conditions, in several tissues or along a time
course, the samples of a donor are correlated. {api}`dream_voom` fits every gene with a
linear mixed model that gives each donor a random intercept, as variancePartition's dream
does [@hoffman2021], in every cell type of a cube, as dreamlet does [@hoffman2026a].
{api}`variance_partitioning` asks the related question of how much of each gene's variance
each variable explains. [Repeated measures and donors](../guides/repeated_measures.ipynb)
compares dream with a fixed donor effect on data, and partitions the variance.

(fixed-or-random-donor)=
## A fixed or a random donor effect

A fixed effect for each donor, the paired design `~ donor + condition`, compares each
donor's samples with each other, but cannot test a variable that is constant within a
donor, such as disease or sex ([A paired design](linear_models.md#paired-design)). A
random intercept for each donor models the correlation of a donor's samples instead, and
keeps such variables testable.

When every donor gives a sample in every condition and every question is within donors,
the paired design and {api}`limma_voom` are the simpler of the two, and the two models find
nearly the same genes, as the guide shows. A variable of the donors needs the random
effect, which also suits donors with unequal numbers of samples. A test between donors
rests on the variation between them: with three donors in each group, dream's tests of a variable of the donors
called too many genes in the guide's simulation, while its tests within donors held their
error rate. dream_voom computes what variancePartition's dream computes
([Checks](#mixed-models-checks)), so this is a property of the method with few donors.
When every donor was sampled alike, compare the profiles pooled by donor
({api}`Cube.collapse_samples`) with {api}`limma_voom` for such a question, and test the
changes within donors with {api}`dream_voom`.

(mixed-model)=
## The model

Every gene is fitted with the same model, in lme4's notation `(1 | donor)` [@bates2015]:

$$
y = X\beta + Z_1 b_1 + \dots + Z_K b_K + e, \qquad
b_k \sim N(0, \sigma_k^2 I), \qquad e \sim N(0, \sigma^2 W^{-1}),
$$

where $X$ holds the fixed effects, $Z_k$ assigns each sample to a level of the $k$-th
grouping factor, one random intercept per level, and $W$ holds the gene's precision
weights. Grouping factors may be crossed, `(1 | donor) + (1 | batch)`, or nested,
`(1 | donor/sample)`. The correlation of two samples of one donor, with no other factor and
equal weights, is $\sigma_1^2 / (\sigma_1^2 + \sigma^2)$, the intra-class correlation.

(mixed-model-reml)=
## REML

{api}`fit_mixed_model` estimates the variance components by restricted maximum likelihood
(REML; @patterson1971), as lme4's `lmer` does, into a {api}`MixedModelFit`. REML estimates
them from the part of the data that the fixed effects do not fit, which removes the
downward bias that estimating the fixed effects gives the maximum likelihood estimates of
variances. Let $\theta_k = \sigma_k / \sigma$, the relative standard deviations, and let
$V = W^{-1} + \sum_k \theta_k^2 Z_k Z_k'$. With $\beta$ and $\sigma^2$ profiled out, the
criterion is a function of the $\theta_k$ alone, lme4's profiled REML deviance, which is

$$
\log\det V + \log\det(X' V^{-1} X) + (n - p)\left(1 + \log\frac{2\pi r^2}{n - p}\right),
$$

with $r^2 = (y - X\hat\beta)' V^{-1} (y - X\hat\beta)$ for $n$ samples and $p$ coefficients.
Newton's method minimizes it with exact derivatives [@harville1977]. A variance component
whose estimate is zero is held exactly at zero, the boundary that lme4 reports as a
singular fit. `method="ml"` maximizes the likelihood instead. The fitted values include
the random effects, at their best linear unbiased predictions [@henderson1975]. Their hat
matrix $H$ gives the approximate residual degrees of freedom
$\operatorname{tr}((I - H)'(I - H))$, which lie between $n - p$, with every component at
zero, and $n$ less the rank of the combined design of fixed and random effects; dream
moderates on them ([below](#dream)).

(satterthwaite)=
## Satterthwaite's degrees of freedom

The t-statistic of a contrast $c$ of the fixed effects is $c'\hat\beta / \sqrt{c' C c}$,
with $C$ the covariance matrix of the estimates. Its distribution is approximated by a t
distribution with Satterthwaite's degrees of freedom [@satterthwaite1946; @giesbrecht1985],

$$
\nu = \frac{2 v^2}{g' A g}, \qquad v = c' C c,
$$

where $g$ is the gradient of $v$ in the variance parameters, and $A$ their asymptotic
covariance matrix: twice the inverse Hessian of the REML deviance.
{api}`fixed_effect_tests` gives them in {api}`FixedEffectTests`, as lmerTest does
[@kuznetsova2017], with exact derivatives where lmerTest takes numerical ones. A component
estimated at zero adds nothing to $g' A g$, and with every component at zero $\nu = n - p$,
the degrees of freedom of the fixed-effects model.

(kenward-roger)=
## Kenward-Roger's method

With few samples the variance components are estimated imprecisely, and $C$, computed as if
they were known, understates the variance of the estimates. Kenward and Roger
[@kenward1997] adjust it. Let $W$ be the inverse of REML's expected information for the
variance parameters $\phi$, on which the covariance of the observations,
$\Sigma = \sum_i \phi_i G_i$, depends linearly, and let
$P_i = -X'\Sigma^{-1} G_i \Sigma^{-1} X$ and
$Q_{ij} = X'\Sigma^{-1} G_i \Sigma^{-1} G_j \Sigma^{-1} X$; then

$$
C_A = C + 2\,C \Big(\sum_{ij} W_{ij} \big(Q_{ij} - P_i C P_j\big)\Big) C .
$$

`fixed_effect_tests(fit, df_method="kenward_roger")` divides each estimate by its standard
error from $C_A$, with Satterthwaite's formula for the degrees of freedom on the expected
information, as pbkrtest computes them [@halekoh2014] and lmerTest's
`ddf = "Kenward-Roger"` reports them. In balanced designs the adjustment vanishes and the
tests are those of the analysis of variance. A variance component at zero still counts
here, which lowers the degrees of freedom below Satterthwaite's.

With precision weights, pbkrtest builds $\Sigma$ without lme4's prior weights while it
takes $C$ from the weighted fit, and dream divides each gene's weights by their mean, so
dream's tests treat every observation as having its gene's average weight.
`kenward_roger_weights="average"`, the default, computes these tests; `"precision"` uses
each observation's own weight, which is Kenward-Roger's method for the weighted model.
With voom's precision weights, `"average"` can give standard errors below the unadjusted
ones, which `"precision"` never does. The cost of each gene grows with the cube of the
number of samples, as in pbkrtest, so that in large studies Kenward-Roger's tests take far
longer than Satterthwaite's ([Timings](#validation-timings) gives measured times).

(mixed-model-f-tests)=
## Several contrasts at once

{api}`fixed_effect_f_test` tests that several contrasts are all zero, such as the
differences between three time points, in a {api}`FixedEffectFTest`. With Satterthwaite's
method, as in lmerTest's `contestMD`, the contrasts are first rotated onto the
eigenvectors of their covariance matrix, so that their estimates are uncorrelated; $F$
averages the squares of their t-statistics and is referred to an $F$ distribution with
the same mean, whose denominator degrees of freedom are $2E/(E-r)$ for
$E = \sum_m \nu_m / (\nu_m - 2)$ over the rotated contrasts [@fai1996; @kuznetsova2017].
With Kenward-Roger's method, as in pbkrtest's `KRmodcomp`, the Wald statistic computed
with $C_A$ is scaled and referred to an $F$ distribution whose first two moments it
matches [@kenward1997]. The numerator degrees of freedom $r$ are the number of
linearly independent contrasts.

(dream)=
## dream

dream is limma-voom with the mixed model in place of the linear model
[@hoffman2021]. {api}`dream_voom` takes, in every cell type, the steps of
[limma-voom](#limma-voom-steps) up to the normalization, then those of variancePartition's
`voomWithDreamWeights`, `dream`, `eBayes` and `topTable`:

1. **Precision weights**, {api}`voom_with_dream_weights`: voom's ([voom](#voom)), with each
   gene's mixed model, fitted by maximum likelihood, in place of the linear model, and the
   trend evaluated at fitted values that include the predicted random effects.
2. **Fits and tests**: {api}`fit_mixed_model` by REML with those weights, and
   {api}`fixed_effect_tests` with dream's degrees of freedom.
3. **Moderation**, {api}`moderated_fixed_effect_tests`: empirical Bayes moderation of each
   gene's residual variance on the approximate residual degrees of freedom of the mixed
   model ([Empirical Bayes moderation](empirical_bayes.md)). The moderated t-statistic has
   the degrees of freedom of its test plus the prior's, and the result
   ({api}`ModeratedFixedEffectTests`) adds a z-statistic with the same p-value, which puts
   tests on different degrees of freedom on one scale.

The degrees of freedom follow dream's default, `df_method="adaptive"`: Kenward-Roger's in
a cell type of fewer than 20 samples and Satterthwaite's from 20 on, as dream chooses
them, although its help page gives the threshold as "<= 20". Where the adjusted covariance
matrix of a gene is not positive definite, or its condition number exceeds 1e6, dream
falls back to Satterthwaite's method for that gene, and so does {api}`dream_voom`; since
the condition number changes with the units of continuous covariates, it helps to centre
and scale them. In each cell type, the mixed model takes its random effects from the
samples that remain, and a cell type whose samples cannot estimate them, such as one with
one sample per donor, is reported and not fitted.

As in variancePartition, the variance that dream_voom moderates is that of an observation
with the gene's average weight. With voom's precision weights, that is roughly the variance
that the trend predicts at the gene's expression level, so the variances moderated still
follow the mean-variance trend that the weights had taken out; `trend=True` lets the prior
follow it too.

`f_tests` gives sets of contrasts, each tested at once by the moderated F-statistic
$t' R^{-1} t / r$ of its moderated t-statistics and the correlation $R$ of their estimates,
as `topTable` gives it for several coefficients. Each gene's F-test has degrees of freedom
of its own, those of $r$ uncorrelated moderated t-statistics combined as
[above](#mixed-model-f-tests), so that with one contrast the F-test and the t-test agree.

(variance-fractions)=
## Variance partitioning

Which variables drive the differences between samples, and which must be adjusted for?
Variance partitioning [@hoffman2016] fits every gene with one mixed model that holds every
variable of the design, the categorical ones as random intercepts (donor, condition, batch)
and the continuous ones as fixed effects (age), and reports the fraction of the gene's
variance that each variable explains:

$$
\frac{\hat\sigma^2_v}{\sum_k \hat\sigma^2_k + \sum_j \hat\sigma^2_{\beta_j} +
\hat\sigma^2},
$$

where a random effect $k$ contributes its variance component $\hat\sigma^2_k$; a
fixed-effect term $j$, the variance over the samples of its part of the fitted values,
$\hat\sigma^2_{\beta_j} = \operatorname{var}(X_j \hat\beta_j)$, and the residual its
variance $\hat\sigma^2$. The fractions of a gene sum to one; without fixed effects, the
fraction of a random effect is the intra-class correlation of two samples that share its
level and no other.

{api}`variance_fractions` computes them into a {api}`VarianceFractions`, as
variancePartition's `fitExtractVarPartModel` does:

- The variances are estimated by maximum likelihood, variancePartition's default, which its
  help page explains: REML can bias the fractions substantially, where maximum likelihood
  gave unbiased ones. `method="reml"` fits by REML.
- With precision weights, such as voom's, each gene's weights are scaled to a mean of one,
  so that the residual variance refers to an observation of average weight, and the
  variances of the fitted fixed effects are weighted ones.
- With random effects, the variance of the fitted fixed effects as a whole is split among
  their terms in proportion to the variance of each term's own contribution. Without random
  effects, the design is fitted by least squares, and each term, factors included, keeps
  the variance of its contribution.
- A variance component at zero, lme4's singular fit, has a fraction of exactly zero. The
  fractions are unaffected by rescaling a gene's weights or values, shifting its values,
  or relabelling the levels of a factor.

{api}`variance_partitioning` takes the steps of {api}`dream_voom` as far as voom's
precision weights in every cell type of a cube, partitions the variance, and records the
samples each fit used in a {api}`VariancePartitioningResult`;
{api}`VariancePartitioningResult.table` gives the fractions of one cell type.
{api}`plot_variance_fractions` draws variancePartition's violins of the fractions over the
genes [@hintze1998], with Silverman's bandwidth [@silverman1986], and
{api}`plot_feature_variance_fractions` its bars for chosen genes. The residual fraction
mixes biological variation, the precision of the measurements and technical variation that
the design does not name, so its size alone does not say whether variables are missing
from the design.

(mixed-models-defaults)=
## Defaults and choices

- **A fixed or a random donor effect.** {api}`limma_voom` with `~ donor + condition` when
  every question is within donors; {api}`dream_voom` for a variable of the donors or
  unequal numbers of samples; the profiles pooled by donor and {api}`limma_voom` for a
  question between donors with few donors in each group ([above](#fixed-or-random-donor)).
- **Degrees of freedom.** `df_method="adaptive"`, dream's default, as above;
  `"satterthwaite"` or `"kenward_roger"` fixes one for every cell type.
  `kenward_roger_weights="average"` computes Kenward-Roger's tests as dream does, and
  `"precision"` with each observation's weight.
- **The span** of the trend of {api}`dream_voom` and {api}`voom_with_dream_weights`: chosen
  from the number of genes, as voom chooses it ([The span](#voom-span)); `span=0.5`
  reproduces variancePartition's `voomWithDreamWeights`.
- **The prior.** A constant prior by default, as in variancePartition; `trend=True` lets it
  follow the expression level, and `robust=True` protects it from genes with outlying
  variances.
- **Weights.** `prior_weights="n_cells"` weights each sample by its number of cells, as
  dreamlet does ([Prior weights from the number of cells](#prior-weights-from-cells)).
- **The genes tested.** `expression_filter={"sample_groups": "condition"}` makes
  dream_voom and limma_voom test the same genes of the same samples ([Filtering by
  expression](#filtering-by-expression)).
- **Variance partitioning.** `method="ml"`, variancePartition's default, or `"reml"`.

(mixed-models-limits)=
## Limits and differences

- **Random intercepts only.** Random slopes, and with them the partition of the variance
  within subsets of the samples, such as `(tissue + 0 | donor)`, are not implemented, nor
  are generalized linear mixed models, bootstrap intervals of the fractions
  (`varPartConfInf`) and sample quality weights for dream ([Not implemented
  yet](#not-implemented-yet)).
- **Complete data.** Every gene needs a value, with a positive weight, in every sample: a
  missing value or a zero weight is an error, where lme4 leaves out a sample whose value
  is missing and gives a zero weight an infinite REML criterion. A design whose fixed and
  random effects, taken together, span every sample fits every gene exactly and is
  refused, although lme4 can fit it.
- **Genes without an optimum.** A gene whose values the fixed and random effects
  reproduce, such as one constant within every donor, has no REML or maximum likelihood
  estimate: the criterion keeps falling as the residual variance shrinks to zero. It is
  left unfitted, with NaN tests and fractions, and out of dream's trend. lme4 warns, or
  fits it with a residual variance near zero, which variancePartition keeps in the trend,
  tests, and gives a donor fraction near one.
- **Local optima.** Like lme4's optimizer, Newton's method finds a local minimum, which is
  not the lowest where the criterion has several.
- **dream's choices.** dream decides which genes fall back to Satterthwaite's method from
  R's estimate of the condition number, and dream_voom from the exact one; they can differ
  within about 2% of 1e6. dream's `topTable` computes the p-values of every gene's F-test
  on the degrees of freedom of the first gene for the first coefficient listed, so that
  they depend on the order of the genes and of the coefficients; here each gene has its
  own. Beyond a p-value of 1e-300, the z-statistic is computed from the logarithm of the
  tail probability, where `topTable`'s stops at 37.07. A design without random effects is
  refused, where dream fits it with limma; dream_voom points to {api}`limma_voom`.
- **Variance partitioning.** With random effects, every categorical variable must be one
  of them, and the formula needs an intercept, as variancePartition requires. The
  variables are listed in the design's order, where variancePartition sorts the random
  effects alphabetically. With `method="reml"` and no random effects, the residual
  variance is REML's, over $n - p$, where variancePartition divides by $n$ whatever its
  `REML` argument says.

(mixed-models-checks)=
## Checks and further reading

The fits and tests are compared with lme4, lmerTest, pbkrtest and variancePartition 1.36.2,
run as separate programs, gene by gene:

- REML fits and Satterthwaite's t-tests against lmerTest 3.2.1 on lme4 2.0.6: close
  (1e-6; 1e-5 for Satterthwaite's degrees of freedom, which lmerTest differentiates
  numerically) ([Validation](#validation-mixed-models)).
- Kenward-Roger's t-tests and F-tests, and Satterthwaite's F-tests, against pbkrtest 0.5.3
  and lmerTest 3.1-3, on lme4 1.1-38 with variancePartition: exact (1e-10) at given
  estimates, and 1e-5 for the degrees of freedom and p-values of Satterthwaite's F-tests,
  which lmerTest differentiates numerically; dream's choice of degrees of freedom and its
  F-tests against variancePartition: close (1e-6)
  ([Validation](#validation-kenward-roger-and-f-tests)).
- dream's weights, fits and moderated tests, and dream_voom on a cube, against
  variancePartition and limma 3.62.1: close (1e-6; 1e-5 for Satterthwaite's degrees of
  freedom and what uses them) ([Validation](#validation-dream)).
- Variance partitioning and its figures against variancePartition: close (1e-6), exact at
  given estimates ([Validation](#validation-variance-partitioning)).

Further reading:

- @hoffman2016 introduced variancePartition, which uses linear mixed models to attribute
  the variation of each gene to disease, sex, tissue, genetics or technical factors.
- @hoffman2021 introduced dream, mixed models for repeated measures designs, and showed
  that other methods can give reproducible false positives driven by the genetic
  regulation of expression.
- @hoffman2026a introduced dreamlet, which runs precision-weighted mixed models on the
  pseudobulk profiles of every cell type, and applied it to 1.4 million nuclei from 150
  Alzheimer's disease cases and 149 controls.

[Composition tests](composition.md), the next chapter, turns from the genes of each cell
type to the shares of the cell types.
