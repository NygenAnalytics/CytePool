---
description: Fixed-effect and random-effects meta-analysis of every gene across studies, as metafor computes it, with the between-study variance, Cochran's Q, I-squared and the Knapp-Hartung adjustment; combined p-values, partial conjunction tests of replication, Storey's pi0 and pi1, and how well studies replicate each other.
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
for every cell type and contrast ({api}`MetaAnalysisResult.table`). Where the studies
give p-values rather than comparable effects, or to ask in how many studies a gene
changes, their p-values are combined, and
{api}`differential_expression_replication` says how well the studies replicate each
other (see [](#combined-p-values) and below).

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

(combined-p-values)=
## Combined p-values

{api}`combine_p_values` combines, for every gene, the p-values $p_1, \dots, p_k$ of the
$k$ studies that tested it into one p-value of the hypothesis that no study has an
effect, as scipy's `combine_pvalues` and the R package metap compute them; it returns a
{api}`CombinedPValues`. With $p_{(1)} \le \dots \le p_{(k)}$ in order:

| Method | Statistic | Its distribution without effects |
| --- | --- | --- |
| `"fisher"` [@fisher1932] | $X = -2 \sum \log p_i$ | $\chi^2_{2k}$, upper tail |
| `"stouffer"` [@stouffer1949; @liptak1958] | $Z = \sum w_i z_i / \sqrt{\sum w_i^2}$, $z_i = \Phi^{-1}(1 - p_i)$ | standard normal, upper tail |
| `"tippett"` [@tippett1931] | $p_{(1)}$ | $1 - (1 - p_{(1)})^k$ |
| `"wilkinson"` [@wilkinson1951] | $p_{(r)}$ | beta with parameters $r$ and $k - r + 1$ |
| `"pearson"` [@pearson1933] | $X = -2 \sum \log (1 - p_i)$ | $\chi^2_{2k}$, lower tail |
| `"mudholkar_george"` [@george1983] | $\sum \log((1 - p_i)/p_i)$, scaled | Student's t on $5k + 4$ degrees of freedom |
| `"simes"` [@simes1986] | $\min_j k\, p_{(j)} / j$ | the p-value itself |

Fisher's and Tippett's methods respond to the strongest study; Stouffer's, the logit and
Pearson's to evidence that the studies share. Stouffer's weights are equal by default;
the square roots of the studies' sample sizes are the usual choice [@zaykin2011]. A
small p-value of Wilkinson's maximum ($r = k$) says that every study has some evidence,
not that every study has an effect: that is the partial conjunction below. A gene that
some studies did not test is combined over the others, and `n_studies` says how many.

The methods are defined for one-sided p-values, for one direction of effect. Two-sided
p-values combined as they are test whether some study has an effect in either
direction, and effects in opposite directions do not cancel. `directions`, the signs of
the studies' effects, such as their log2 fold changes, turns two-sided p-values into
one-sided ones, $p/2$ for an effect above zero and $1 - p/2$ below, as metap's
`two2one`; each direction is combined, and the p-value is
$\min(1, 2 \min(p_\text{up}, p_\text{down}))$ [@owen2009], which for Stouffer's method
is the two-sided p-value of $Z$. The one-sided p-value near 0 is kept exactly, so
p-values near 1e-300 keep their digits. p-values of 0 and 1 are combined as the limits
they are, where metap leaves them out; one study gives its own p-value.

## Replicated in at least u studies

One study's effect makes the global hypothesis false, so a small combined p-value does
not say that an effect replicates. {api}`partial_conjunction` tests the hypothesis that
fewer than $u$ of the $n$ studies that tested a gene have an effect by combining its
$n - u + 1$ largest p-values, $p_{(u)}, \dots, p_{(n)}$ [@benjamini2008]: by Simes's
method, $\min_{i \ge u} (n - u + 1)\, p_{(i)} / (i - u + 1)$, or Fisher's or Stouffer's
(unweighted) on $n - u + 1$ p-values. For $u = 1$ it is the global test, and for
$u = n$ every method gives $p_{(n)}$. With `directions`, the effects must share their
direction. Benjamini and Hochberg's adjustment of these p-values over the genes
controls the false discovery rate of the claims that a gene has an effect in at least
$u$ studies [@benjamini2008].

## The proportion of true nulls and pi1

Among $m$ tests, null p-values are uniform, so the p-values at least $\lambda$ hold
about $\pi_0 m (1 - \lambda)$ null ones, and
$\hat\pi_0(\lambda) = \#\{p \ge \lambda\} / (m (1 - \lambda))$ estimates the
proportion of true nulls [@storey2002]. {api}`proportion_of_true_nulls` gives it as
qvalue's `pi0est` does, with its defaults and options, and returns a
{api}`TrueNullProportion`: at one $\lambda$; smoothed over
$\lambda = 0.05, 0.10, \dots, 0.95$ by a cubic smoothing spline with 3 degrees of
freedom, read at the largest $\lambda$ [@storey2003]; or at the $\lambda$ that
minimizes a bootstrap estimate of its mean squared error [@storey2004]. The details that
qvalue's documentation leaves open were found by running it: it counts the p-values at
least $\lambda$, where the paper counts those above; its smoother is R's
`smooth.spline`, whose search for the smoothing parameter CytePool follows step by step;
and its bootstrap measures the bias from the 10% quantile of the
$\hat\pi_0(\lambda)$. $\pi_1 = 1 - \pi_0$ of one study's p-values among the genes
another study finds is the share of them with an effect in it, which counts effects too
small for the second study to find on its own [@storey2003].

## Replication across the studies of a collection

{api}`differential_expression_replication` reads one cell type and contrast of the
studies of a {api}`CubeCollection`, lined up by {api}`align_study_results`, and returns
a {api}`ReplicationResult` with four tables:

- `genes`: for every gene, the studies that tested it and find it (adjusted p-value
  below `alpha`), up and down; the combined p-value, by Stouffer's method weighted by
  the square roots of the studies' samples and signed by their log2 fold changes; the
  partial conjunction p-value of an effect in at least `at_least` studies in one
  direction (Simes's by default), both adjusted over the genes; the share of the
  studies whose effect has the combined direction; and Cochran's $Q$, $I^2$ and
  $\tau^2$ of its log2 fold changes, from {api}`meta_analysis`;
- `study_pairs`: for every ordered pair of studies, $\pi_1$ and the sign agreement of
  the genes the first finds in the second, and how many the second finds too;
- `found_in`: the genes that at least $k$ studies find, and in one direction;
- `studies`: each study's samples, weight, and the genes it tested and finds.

Most genes that one study finds are found by no other [@nakatsuka2025], and these
tables say how far that holds. {api}`plot_study_agreement` draws $\pi_1$ or the sign
agreement of every pair of studies, and {api}`plot_replicated_genes` the genes that
each number of studies finds.

## Limits

The studies are taken as independent, with known sampling variances. Meta-regression on
study-level covariates, multivariate meta-analysis of correlated effects, and shrinkage
across conditions such as mash [@urbut2019] are not implemented. Combined p-values
assume independent studies whose p-values are uniform without effects; for discrete
tests they are conservative. The r-values of a primary study and its follow-up
[@heller2014] are not implemented. metap's methods of Edgington (sums of p-values),
Lancaster (inverse chi-squared), the inverse t, the mean z and vote counting are left
out. Held-out prediction of case status from the genes of several studies
[@nakatsuka2025] is not implemented yet.
