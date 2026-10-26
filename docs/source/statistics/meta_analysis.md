---
description: Fixed-effect and random-effects meta-analysis of every gene across studies, as metafor computes it, with the between-study variance, Cochran's Q, I-squared and the Knapp-Hartung adjustment.
---

# Meta-analysis

A single study often finds differences that others do not reproduce: of the genes called
differentially expressed in Alzheimer's disease in one dataset, over 85% were not
reproduced in any of 16 others [@nakatsuka2025]. A meta-analysis combines the estimates
of several studies into one, weighs each by its precision, and measures how much the
studies disagree. {api}`meta_analysis` does it for every gene at once, as metafor's
`rma.uni` does for one effect without moderators [@viechtbauer2010], and
{api}`differential_expression_meta_analysis` does it for the results of
{api}`limma_voom` or {api}`limma_trend` on several cubes, matched by cell type, contrast
and gene id. {api}`meta_analysis` returns a {api}`MetaAnalysisFit`, and
{api}`differential_expression_meta_analysis` a {api}`MetaAnalysisResult`, with a table
for every cell type and contrast ({api}`MetaAnalysisResult.table`).

## The model

Study $i$ estimates a gene's effect by $y_i$, with a standard error $s_i$ whose square,
$v_i$, is taken as known. In the random-effects model the true effects of the studies
vary around a mean $\mu$ with the between-study variance $\tau^2$ [@dersimonian1986]:

$$
y_i = \mu + u_i + e_i, \qquad u_i \sim N(0, \tau^2), \qquad e_i \sim N(0, v_i).
$$

With the weights $w_i = 1 / (v_i + \tau^2)$, $\hat\mu = \sum w_i y_i / \sum w_i$ with
standard error $1 / \sqrt{\sum w_i}$. The fixed-effect model, one true effect for every
study, is the case $\tau^2 = 0$: the inverse-variance weighted mean.

## The between-study variance

- `"reml"`, metafor's default: the restricted maximum likelihood estimate
  [@viechtbauer2005]. Where the restricted likelihood has two maxima, as when an
  imprecise study lies far from precise ones that agree, CytePool takes the larger one;
  metafor's Fisher scoring can stop at the smaller one.
- `"dl"`: DerSimonian and Laird's method of moments,
  $\max(0, (Q - (k - 1)) / (\sum w_i - \sum w_i^2 / \sum w_i))$ with $w_i = 1 / v_i$.
- `"pm"`: Paule and Mandel's estimate [@paule1982], at which the generalized Q statistic
  equals its expectation $k - 1$.
- `"fixed"`: $\tau^2 = 0$.

## Tests and heterogeneity

The default z-test refers $\hat\mu / \mathrm{SE}$ to the normal distribution.
`test="knapp_hartung"` multiplies the standard error by $\sqrt{q}$, with
$q = \sum w_i (y_i - \hat\mu)^2 / (k - 1)$, and uses a t distribution on $k - 1$ degrees
of freedom [@hartung1999; @sidik2002; @knapp2003], which metafor's documentation
recommends for few studies.

Cochran's $Q = \sum (y_i - \hat\mu_{\text{fixed}})^2 / v_i$ tests whether the studies
agree [@cochran1954]. $I^2$ is the share of the variation that comes from heterogeneity
rather than sampling error, and $H^2$ the ratio of the total to the sampling variance
[@higgins2002]; CytePool reports $I^2$ as a proportion, where metafor reports a
percentage.

## Genes that some studies did not measure

NaN marks a study that did not measure or test a gene. Each gene is analysed over the
studies that measured it; {api}`differential_expression_meta_analysis` analyses a gene
tested by at least `min_studies` studies (two by default, since one study says nothing
about $\tau^2$). It takes limma's moderated standard errors as the known sampling
errors.

A gene missing from a study's results was not measured there, or was measured and left
out, as by the expression filter, which are different findings. For the studies of a
{api}`CubeCollection`, each analysed on its aligned cube,
{api}`align_study_results` lines up the results of one cell type and contrast with a
row for every study and gene, which says whether the study tested the gene and, where
it did not, why: it did not measure the gene, measured it and left it out, or did not
analyse the cell type or the contrast.

## Limits

The studies are taken as independent, with known sampling variances. Meta-regression on
study-level covariates, multivariate meta-analysis of correlated effects, and shrinkage
across conditions such as mash [@urbut2019] are not implemented.
