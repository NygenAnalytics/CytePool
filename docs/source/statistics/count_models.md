---
description: The negative binomial models behind edgeR's quasi-likelihood F-tests and DESeq2's Wald and likelihood ratio tests, as CytePool computes them, with the normalization they rest on.
---

# Count models

limma models log counts; edgeR and DESeq2 model the counts themselves, with a negative
binomial distribution whose variance grows with the mean. CytePool implements edgeR 4's
quasi-likelihood pipeline and DESeq2's analysis, each checked against the R package on
the same data.

## Normalization

Counts depend on how deeply each sample was sequenced and on its composition: a few very
highly expressed genes in one sample take reads from all others. Two corrections are
standard. edgeR multiplies each library size by a normalization factor, from the trimmed
mean of M values ({api}`tmm_factors`; @robinson2010a), the upper quartile
({api}`upper_quartile_factors`; @bullard2010) or the relative log expression
({api}`rle_factors`; @anders2010). DESeq2 replaces library sizes by size factors, the
median of each sample's ratios to the geometric mean of every gene
({api}`median_ratio_size_factors`; @anders2010). {api}`cpm` and {api}`log_cpm` give
counts per million, and {api}`filter_by_expression` keeps genes with enough counts in
enough samples for the design [@chen2016].

## The negative binomial model

For the count $y_{gi}$ of gene $g$ in sample $i$,

$$
y_{gi} \sim \operatorname{NB}(\mu_{gi}, \phi_{gi}), \qquad
\operatorname{var}(y_{gi}) = \mu_{gi} + \phi_{gi} \mu_{gi}^2, \qquad
\log \mu_{gi} = x_i' \beta_g + o_{gi},
$$

with $\phi$ the dispersion and the offset $o_{gi}$ usually the log of the effective
library size [@mccarthy2012]. $\sqrt{\phi}$ is the biological coefficient of variation.
The coefficients minimize the deviance

$$
D_g = \sum_i w_{gi}\, d(y_{gi}, \mu_{gi}, \phi_{gi}), \qquad
d(y, \mu, \phi) = 2\left[y \log\frac{y}{\mu} - \left(y + \frac{1}{\phi}\right)
\log\frac{1 + \phi y}{1 + \phi \mu}\right],
$$

which {api}`fit_negative_binomial_model` finds for every gene, as edgeR's `glmFit` does
({api}`NegativeBinomialFit`), and which {api}`negative_binomial_deviance` and
{api}`negative_binomial_unit_deviance` compute exactly [@dunn2018].

## edgeR's quasi-likelihood F-tests

The quasi-likelihood model gives each gene a second, gene-wise dispersion $\sigma_g^2$
on top of a negative binomial dispersion $\phi$ shared by all genes [@lund2012]:

$$
\operatorname{var}(y_{gi}) = \sigma_g^2 (\mu_{gi} + \phi \mu_{gi}^2).
$$

{api}`fit_quasi_likelihood` follows edgeR 4's defaults [@chen2025]:

1. $\phi$ is the Cox-Reid common dispersion of the most highly expressed genes
   ({api}`common_dispersion`): the maximum over $\phi$ of the summed adjusted profile
   log-likelihoods $\ell_g(\phi) - \tfrac{1}{2}\log\det(X' W_g X)$ [@cox1987;
   @mccarthy2012].
2. Each gene's quasi-dispersion is its deviance over its residual degrees of freedom,
   both adjusted for small counts: each unit deviance is approximated by a scaled
   chi-squared variable with the exact mean and variance of the unit deviance
   ({api}`negative_binomial_unit_deviance_moments`, which gives
   {api}`UnitDevianceMoments`), where edgeR approximates them.
3. The quasi-dispersions are moderated by empirical Bayes towards a trend in the average
   log CPM ({api}`average_log_cpm`), with limma's estimator for unequal degrees of
   freedom ([Empirical Bayes moderation](empirical_bayes.md)).

The fit is a {api}`QuasiLikelihoodFit`. {api}`quasi_likelihood_f_test` fits the model
without the tested coefficients or contrasts, and divides the likelihood ratio statistic
by its degrees of freedom and by the posterior quasi-dispersion; its
{api}`QuasiLikelihoodTest` has edgeR's columns: log fold change, log CPM, F, p-value and
FDR. The [edgeR User's
Guide](https://bioconductor.org/packages/release/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf)
(section 1.4) recommends these F-tests for bulk RNA-seq because they account for the
uncertainty of the dispersion estimates, which gives stricter error rate control.

## DESeq2

DESeq2 models the same distribution on the log2 scale with size factors $s_i$,
$\log_2 \mu_{gi} = \log_2 s_i + x_i' \beta_g$, and shrinks the dispersions in four steps
[@love2014], which {api}`deseq2_dispersions` reproduces ({api}`DESeq2Dispersions`):

1. gene-wise estimates, the maxima of the Cox-Reid adjusted likelihood;
2. a trend in the mean normalized count $m$, $\alpha_{tr}(m) = a_0 + a_1 / m$, fitted by
   a gamma regression;
3. a normal prior on the log dispersions around the trend, with its variance from the
   spread of the gene-wise estimates;
4. maximum a posteriori estimates, keeping the gene-wise estimate of genes far above the
   trend (dispersion outliers).

{api}`fit_deseq2` fits every gene at those dispersions ({api}`DESeq2Fit`).
{api}`deseq2_wald_test` tests a coefficient or contrast $c$ by
$c'\hat\beta / \sqrt{c'\Sigma c}$ against the normal distribution;
{api}`deseq2_likelihood_ratio_test` tests a reduced design by the difference of the
deviances against a chi-squared distribution, which tests several coefficients at once,
such as every level of a factor or every time point. Both report DESeq2's `results()`
columns ({api}`DESeq2Results`), filter genes with low mean counts before adjusting the
p-values (independent filtering; @bourgon2010) and remove the p-values of genes with an
influential outlier count (Cook's distance; @cook1977).

## What differs from the R packages

CytePool's fits converge to the maximum where edgeR's and DESeq2's stop at a tolerance,
and its deviances and unit deviance moments are exact where edgeR approximates them; the
differences are reported, with their sizes, on the [Validation](../validation.md) page.
DESeq2's replacement of outlier counts, shrunken fold changes (`lfcShrink`) and edgeR's
`legacy = TRUE` methods are not implemented.
