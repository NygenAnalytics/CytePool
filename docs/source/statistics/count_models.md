---
description: How edgeR's quasi-likelihood F-tests and DESeq2's Wald and likelihood ratio tests find the genes that change in one matrix of counts, from negative binomial models of every gene.
---

# edgeR and DESeq2

[limma-voom and limma-trend](limma_voom.md) model log-CPM values. edgeR and DESeq2 model
the counts themselves, with a negative binomial distribution whose variance grows with the
mean, and they are the alternatives for one matrix of counts such as bulk RNA-seq, where
{api}`limma_voom` on a cube of one group ({api}`from_bulk`) stays the default.
{api}`fit_quasi_likelihood` and {api}`quasi_likelihood_f_test` run edgeR's quasi-likelihood
F-tests [@chen2025], and {api}`fit_deseq2` with {api}`deseq2_wald_test` or
{api}`deseq2_likelihood_ratio_test` runs DESeq2 [@love2014]. [Bulk
RNA-seq](../guides/bulk_rna_seq.ipynb) runs both, with limma-voom, on a simulated experiment
whose true changes are known.

(one-matrix-of-counts)=
## One matrix of counts

Bulk RNA-seq, or one cell type taken out of a cube ({api}`Cube.layer`), is a samples x genes
matrix of counts. limma, edgeR and DESeq2 all test it, and they agree on genes with clear
effects. All three moderate the variability of each gene by borrowing from the others; they
differ in the scale they model, log-CPM values or counts, and in how they shrink: limma
moderates residual variances towards a prior ([Empirical Bayes
moderation](empirical_bayes.md)), edgeR moderates quasi-dispersions, and DESeq2 shrinks
dispersions towards a trend in the mean count.

The choice rests mostly on the design and on what comes next. edgeR's User's Guide (section
1.4) recommends its quasi-likelihood F-tests for bulk RNA-seq, because they account for the
uncertainty of the dispersion estimates, which gives stricter control of the error rate.
In DESeq2, the likelihood ratio test asks about several coefficients together, all the
levels of a factor or all the time points, which its Wald test takes one at a time;
edgeR's F-tests and limma's moderated F-tests also test several at once. limma's tables
carry moderated standard errors, which a [meta-analysis](meta_analysis.md) combines across
studies.

(negative-binomial-model)=
## The negative binomial model

For the count $y_{gi}$ of gene $g$ in sample $i$,

$$
y_{gi} \sim \operatorname{NB}(\mu_{gi}, \phi_{gi}), \qquad
\operatorname{var}(y_{gi}) = \mu_{gi} + \phi_{gi} \mu_{gi}^2, \qquad
\log \mu_{gi} = x_i' \beta_g + o_{gi},
$$

with $\phi$ the dispersion and the offset $o_{gi}$ usually the log of the effective library
size ([Normalization factors and size factors](#normalization-and-size-factors);
@mccarthy2012). $\sqrt{\phi}$ is the biological coefficient of variation: edgeR's User's Guide
gives typical values of 0.4 for human samples, 0.1 for genetically identical model organisms
and 0.01 for technical replicates. The coefficients minimize the deviance

$$
D_g = \sum_i w_{gi}\, d(y_{gi}, \mu_{gi}, \phi_{gi}), \qquad
d(y, \mu, \phi) = 2\left[y \log\frac{y}{\mu} - \left(y + \frac{1}{\phi}\right)
\log\frac{1 + \phi y}{1 + \phi \mu}\right],
$$

with prior weights $w_{gi}$. {api}`fit_negative_binomial_model` finds the coefficients of
every gene at dispersions that are given, as edgeR's `glmFit` does
({api}`NegativeBinomialFit`), and {api}`negative_binomial_unit_deviance` and
{api}`negative_binomial_deviance` compute the unit deviances and their sums exactly
[@dunn2018]. A one-way layout, a design with as many distinct rows as columns such as one
mean per group, is fitted one group at a time. Newton's method fits the other designs,
started as edgeR starts it, from the fit of a single mean to all the samples, with steps
damped where they would raise the deviance [@levenberg1944; @marquardt1963]. The
log-likelihood is concave in the coefficients, so its maximum is unique when it exists.

It does not exist where the fitted values of some samples can fall towards zero while the
others stay put, as when every count of a group is zero: the fitted values of those samples
tend to zero, and their coefficients grow without bound. The fits stop with such
coefficients large but finite; in a one-way layout, a group without counts gets a log mean
of $-10^8$, as in edgeR. edgeR's log fold changes, and those of these fits, come from a
second fit after an average count of 0.125, edgeR's default, is added to every count,
which shrinks them towards zero and keeps them finite [@phipson2013]; the tests do not use
them.

(cox-reid)=
### The Cox-Reid adjusted profile likelihood

edgeR and DESeq2 both estimate dispersions by maximizing, over $\phi$, the adjusted profile
log-likelihood of a gene,

$$
\ell_g(\phi, \hat\beta_g(\phi)) - \tfrac12 \log\det(X' W_g X), \qquad
W_g = \operatorname{diag}\!\left(\frac{\mu_{gi}}{1 + \phi \mu_{gi}}\right),
$$

where $\hat\beta_g(\phi)$ maximizes the likelihood at that $\phi$; DESeq2 holds it at a
first fit instead [@cox1987; @mccarthy2012]. The adjustment, which a REML criterion
parallels, removes most of the downward bias that estimating the coefficients gives the
maximum likelihood estimate of the dispersion [@robinson2008]. {api}`common_dispersion`
maximizes the sum over the genes, edgeR's common dispersion; DESeq2 maximizes each gene's
own.

(quasi-likelihood)=
## edgeR's quasi-likelihood F-tests

The quasi-likelihood model gives each gene a second, gene-wise dispersion $\sigma_g^2$, the
quasi-dispersion, on top of a negative binomial dispersion $\phi$ shared by all genes
[@lund2012]:

$$
\operatorname{var}(y_{gi}) = \sigma_g^2 (\mu_{gi} + \phi \mu_{gi}^2).
$$

$\phi$ carries the variance common to the genes, and $\sigma_g^2$ the variation between
genes, which empirical Bayes moderates as limma moderates residual variances.
{api}`fit_quasi_likelihood` takes the steps of edgeR 4.4's `glmQLFit` with its defaults
(`legacy = FALSE`; @chen2025):

1. **Abundance.** Each gene's average log-CPM ({api}`average_log2_cpm`, edgeR's
   `aveLogCPM`, with an average count of 2 added) measures its expression: it orders the
   genes for step 2, and the trends of steps 4 and 5 follow it.
2. **The dispersion.** Unless it is given, $\phi$ is the Cox-Reid common dispersion
   ([above](#cox-reid)) of the most highly expressed genes, a share that is smaller the
   more genes and residual degrees of freedom there are.
3. **Quasi-dispersions.** Each gene's fit uses the working dispersion $\phi / A$, $A$
   coming from step 4. edgeR 4 treats a count's unit deviance as $s\,\chi^2_k$, a
   chi-squared variable on $k$ degrees of freedom scaled by $s$, with the unit deviance's
   mean $E$ and variance $V$: $s = V / (2E)$ and $k = 2E^2 / V$
   ({api}`negative_binomial_unit_deviance_moments`, which gives them in a
   {api}`UnitDevianceMoments`). The unit deviance over $s$ then carries $k (1 - h)$
   degrees of freedom, $h$ being the count's leverage. Summed over the samples, these give
   each gene an adjusted deviance and fractional degrees of freedom, whose ratio is its
   quasi-dispersion. The adjustment matters for small counts, where a unit deviance is far
   from chi-squared on one degree of freedom.
4. **The average quasi-dispersion** $A$, which "scales the NB dispersion", as edgeR's help
   page says. It starts at 1 and is computed again twice, from fits at the current
   $\phi / A$: the 0.9 quantile, over the genes, of the trend that smooths the fourth roots
   of their quasi-dispersions against the abundance, raised back to the fourth power, or 1
   if that is larger.
5. **Moderation.** The quasi-dispersions are moderated by empirical Bayes towards a trend
   in the abundance, by the estimator that limma uses where degrees of freedom differ
   ([Estimating the prior](#estimating-the-prior)), robustly with `robust=True`
   [@phipson2016].

The fit is a {api}`QuasiLikelihoodFit`. {api}`quasi_likelihood_f_test` fits the model
without the tested coefficients, or with the tested contrasts at zero, at the same working
dispersion, and divides the likelihood ratio statistic by its degrees of freedom $r$ and by
the posterior quasi-dispersion. The F-statistic has $r$ and $d_0 + d_g$ degrees of freedom,
the prior's and the gene's adjusted ones, capped at the total of the genes' residual
degrees of freedom. Its {api}`QuasiLikelihoodTest` has the columns of edgeR's `topTags`:
log fold change, log-CPM, F, p-value and the false discovery rate of Benjamini and
Hochberg.

(deseq2)=
## DESeq2

DESeq2 models the same distribution on the log2 scale, with the size factors $s_i$ in place
of the library sizes, $\log_2 \mu_{gi} = \log_2 s_i + x_i' \beta_g$, and one dispersion
$\alpha_g$ per gene [@love2014]. {api}`deseq2_dispersions` estimates the dispersions as
`estimateDispersions()` does ({api}`DESeq2Dispersions`):

1. **Gene-wise estimates**: the maxima of the Cox-Reid adjusted likelihood
   ([above](#cox-reid)), conditional on means fitted at a rough dispersion.
2. **A trend** $\alpha_{tr}(m) = a_0 + a_1 / m$ in each gene's mean normalized count $m$,
   fitted by a gamma regression to the gene-wise estimates.
3. **A prior**: the log dispersions are normal around the log trend, with a variance
   estimated from the spread of the gene-wise estimates about it, less the part that
   sampling explains.
4. **Maximum a posteriori estimates**, which shrink each gene towards the trend; a gene
   whose gene-wise estimate lies far above the trend, a dispersion outlier, keeps it.

{api}`fit_deseq2` then fits each gene at its dispersion, as `DESeq()` does
({api}`DESeq2Fit`). {api}`deseq2_wald_test` refers $c'\hat\beta / \sqrt{c' \Sigma c}$, for
a coefficient or contrast $c$, to the normal distribution;
{api}`deseq2_likelihood_ratio_test` tests a reduced design by the difference of the
deviances against a chi-squared distribution, which tests several coefficients at once.
Both report the columns of DESeq2's `results()` ({api}`DESeq2Results`). Before the p-values
are adjusted, the genes of low mean count are filtered out, at the lowest threshold whose
number of genes below `alpha` comes close to the most that any threshold gives
(independent filtering; @bourgon2010), and a gene whose count in one sample is so
influential that its Cook's distance exceeds the 0.99 quantile of an F distribution loses
its p-value [@cook1977].

(count-models-defaults)=
## Defaults and choices

- **Which method.** {api}`limma_voom` on a cube of one group by default; edgeR's
  quasi-likelihood F-tests for the stricter error control that edgeR's User's Guide
  describes; DESeq2's Wald tests, or its likelihood ratio test for several coefficients
  at once ([One matrix of counts](#one-matrix-of-counts)).
- **The genes and the factors.** {api}`fit_quasi_likelihood` takes the counts it is
  given, so filter them by expression and pass TMM factors, as edgeR's workflow does
  ([Normalization and filtering](normalization_and_filtering.md)); {api}`fit_deseq2`
  takes every gene, with DESeq2's size factors by default, and filters on the mean count
  when it adjusts the p-values.
- **The dispersion of the quasi-likelihood fit.** Estimated from the data by default. One
  that is given should come from a model of all the genes, common or trended, and not be
  estimated gene by gene, as edgeR's help page advises, since the variation between genes
  is for the quasi-dispersions to carry.
- **The prior of the quasi-dispersions.** A trend in the abundance by default
  (`trend=True`), or a constant (`False`), or a trend in another covariate; `robust=True`
  protects it from genes with outlying quasi-dispersions, as edgeR's User's Guide
  recommends.
- **DESeq2's trend.** `trend="parametric"`, DESeq2's default, or `"mean"`, the mean of the
  gene-wise estimates.
- **DESeq2's tests.** `alpha` (0.1, as in `results()`) sets the false discovery rate that
  the independent filter is chosen for, `independent_filtering=False` keeps every gene,
  and `cooks_cutoff` changes or turns off the removal of p-values by Cook's distance.

(count-models-limits)=
## Limits and differences

- **One matrix at a time.** The fits run on one matrix of counts, such as one cell type
  taken out of a cube ({api}`Cube.layer`); their tests of every cell type of a cube, and
  the methods of edgeR and DESeq2 that [Not implemented yet](#not-implemented-yet)
  lists, are not implemented.
- **Dispersions as given.** {api}`fit_negative_binomial_model` takes the dispersions as
  given, and {api}`common_dispersion` estimates one common to every gene.
- **Exact deviances and moments.** The deviances are exact. edgeR adds 1e-8 to every
  count and mean, and approximates the unit deviance at small dispersions and at large
  dispersions times means; it also approximates the mean and variance of the unit
  deviance, which are exact here, so that adjusted deviances, quasi-dispersions and
  F-statistics differ slightly from edgeR's.
- **Fits that reach the maximum.** The fits converge to the maximum of the likelihood,
  by steps that use its exact curvature, the observed information, where edgeR's
  documentation describes Fisher scoring, which uses the expected information and has
  the same maximum. edgeR's one-way fits agree with them; its fits of other designs stop
  short of the maximum, the more so where dispersions and weights differ between
  observations, and leave fitted values that tend to zero between about 1e-8 and 1e-7,
  far above CytePool's, so that the log fold changes of such designs differ slightly. In
  a design with an intercept, a gene without counts gets coefficients that make its log
  mean $-10^8$, as edgeR's one-way fit does, where edgeR's other fits return missing
  values.
- **Infinite inputs.** Infinite dispersions or offsets, on which edgeR's fits run without
  end, are refused.
- **The robust prior.** Genes without residual degrees of freedom, such as genes without
  counts, are left out of the test for outlying quasi-dispersions, where limma counts them
  as zero variances ([Limits of empirical Bayes moderation](#empirical-bayes-limits)).
- **Average log-CPM with offsets per observation.** edgeR's `aveLogCPM` then gives the
  first gene a value that ignores its offsets, and the value enters the ranking, the
  trends and that gene's log-CPM; {api}`average_log2_cpm` uses every gene's offsets.
- **DESeq2's estimates.** CytePool finds the maximum of each dispersion's objective, where
  DESeq2's line searches stop once an iteration gains less than 1e-6, and can stall where
  they start. DESeq2's fits stop once the deviance changes by under 1e-8 of its value,
  short of the fixed point for genes of large dispersion, while every fit here
  converges. With three or fewer residual degrees of freedom, the prior's variance is the
  maximum likelihood estimate of the model of the log residuals, the limit of DESeq2's
  criterion as its densities of simulated residuals become exact: DESeq2 chooses it from
  a grid of 1,000 values on [0, 8] by matching such densities, so that its choice moves
  with the simulated sample.
- **Where DESeq2's trend fails.** When the parametric trend has a coefficient that is not
  positive, or does not converge in 11 fits, DESeq2 substitutes a local regression, which
  is not implemented here; CytePool falls back to the mean trend, with a warning, as
  PyDESeq2 does.
- **Outlier counts are not replaced.** The results are those of `DESeq(...,
  minReplicatesForReplace = Inf)`, which equal DESeq2's defaults where no row of the design
  occurs seven times or more.

(count-models-checks)=
## Checks and further reading

Each function is compared with edgeR 4.4.0 or DESeq2 1.46.0, run as separate programs on
simulated counts with genes and groups without counts:

- Negative binomial fits and deviances: close (1e-6) against edgeR's `glmFit`, and against
  the maximum where edgeR stops short; unit deviances against 150-digit arithmetic
  ([Validation](#validation-negative-binomial-fits)).
- Quasi-likelihood F-tests: close (1e-6) where both programs compute the same quantity,
  and within 2e-2 where edgeR approximates the moments of the unit deviance or its fits stop
  short ([Validation](#validation-quasi-likelihood)).
- DESeq2's dispersions, fits and tests, each stage given DESeq2's own inputs: close (1e-6;
  1e-5 for what is computed from DESeq2's fits), with PyDESeq2 0.5.4 for comparison
  ([Validation](#validation-deseq2-tests)).

Further reading:

- @mccarthy2012 introduced generalized linear models of counts with the Cox-Reid
  dispersion for multifactor RNA-seq experiments.
- @lund2012 introduced quasi-likelihood tests with shrunken dispersions for RNA-seq.
- @chen2025 described edgeR 4, whose quasi-likelihood method adjusts deviances for small
  counts.
- @love2014 introduced DESeq2's shrinkage of dispersions and fold changes.
- @bourgon2010 showed that filtering genes on a statistic independent of the test raises
  the number of discoveries.

[Mixed models and dream](mixed_models.md), the next chapter, allows for samples of the
same donor.
