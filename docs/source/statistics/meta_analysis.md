---
description: What several studies say together about each gene, by a fixed-effect or random-effects meta-analysis of their estimates, with the between-study variance and the heterogeneity of the studies.
---

# Meta-analysis

[Aligning studies](aligning_studies.md) analysed each study on its own and lined up the results gene by gene. A meta-analysis combines, for every gene, the studies' estimates of one effect into one, weighs each study by its precision, and measures how far the studies disagree. {api}`meta_analysis` does it for the results of {api}`limma_voom` or {api}`limma_trend` on several studies, a {api}`StudyResults`, as metafor's `rma.uni` does for one effect without moderators [@viechtbauer2010], and {api}`fit_meta_analysis` does it for arrays of effects and standard errors, into a {api}`MetaAnalysisFit`. [Align and combine studies](../guides/combine_studies.ipynb) runs it on six simulated cohorts whose true changes are known.

(meta-analysis-model)=
## The model

Study $i$ estimates a gene's effect by $y_i$, with a standard error $s_i$ whose square, $v_i$, the analysis takes as known. The random-effects model lets the true effects of the studies vary around a mean $\mu$, with the between-study variance $\tau^2$ [@dersimonian1986]:

$$
y_i = \mu + u_i + e_i, \qquad u_i \sim N(0, \tau^2), \qquad e_i \sim N(0, v_i).
$$

Each study weighs $w_i = 1 / (v_i + \tau^2)$ in the estimate $\hat\mu = \sum_i w_i y_i / \sum_i w_i$, whose standard error is $1 / \sqrt{\sum_i w_i}$. The fixed-effect model, one true effect for every study, is the case $\tau^2 = 0$: the inverse-variance weighted mean (metafor's equal-effects model). {api}`meta_analysis` takes each study's log2 fold changes and standard errors, matched by cell type, contrast and gene id, and returns a {api}`MetaAnalysisResult` with a table for every cell type and contrast ({api}`MetaAnalysisResult.table`). {api}`plot_forest` draws one gene's estimate in every study with the pooled estimate, as metafor's `forest` does [@lewis2001].

(between-study-variance)=
## The between-study variance

`method` chooses the estimator of $\tau^2$, over the $k$ studies that tested a gene:

- `"reml"`, metafor's default: the restricted maximum likelihood estimate [@viechtbauer2005]. Where the restricted likelihood has two maxima, as when an imprecise study lies far from precise ones that agree, the estimate is the maximum with the higher likelihood.
- `"dl"`: DerSimonian and Laird's method of moments, $\max(0, (Q - (k - 1)) / (\sum_i w_i - \sum_i w_i^2 / \sum_i w_i))$, for $w_i = 1 / v_i$ and Cochran's $Q$ below.
- `"pm"`: Paule and Mandel's estimate [@paule1982], the $\tau^2$ that makes the generalized $Q$ statistic, $\sum_i w_i (y_i - \hat\mu)^2$, equal to its expectation, $k - 1$, or zero where none does.
- `"fixed"`: $\tau^2 = 0$, the fixed-effect model.

(heterogeneity)=
## Tests and heterogeneity

The default z-test refers $\hat\mu / \mathrm{SE}$ to the normal distribution. `test="knapp_hartung"` multiplies the standard error by $\sqrt{q}$, where $q = \sum_i w_i (y_i - \hat\mu)^2 / (k - 1)$, and uses a t distribution on $k - 1$ degrees of freedom [@hartung1999; @sidik2002; @knapp2003], which metafor's documentation recommends for few studies. Where the studies agree more closely than their standard errors imply, $q < 1$, and the adjusted standard error is the smaller.

Cochran's $Q = \sum (y_i - \hat\mu_{\text{fixed}})^2 / v_i$ tests whether the studies agree [@cochran1954]. $I^2$ is the share of the variation that comes from heterogeneity rather than sampling error, and $H^2$ the ratio of the total to the sampling variance [@higgins2002], computed from the estimate of $\tau^2$, as metafor computes them. $I^2$ is reported as a proportion, where metafor reports a percentage.

(untested-genes)=
## Genes that some studies did not test

A gene that a study did not test has a NaN estimate there, and is meta-analysed over the studies that did. {api}`meta_analysis` analyses a gene that at least `min_studies` studies tested (two by default, since one study says nothing about $\tau^2$); with `min_studies=1`, a gene that one study tested keeps that study's estimate and its z-test, as metafor gives it for one study. {api}`StudyResults.table` says why a study did not test a gene ([One analysis per study](#one-analysis-per-study)).

Each study's standard errors are those of its moderated t-statistics, from limma's empirical Bayes moderation. Like any sampling variance in a meta-analysis, they are taken as known, without their degrees of freedom, so that the z-test of a gene that only one study tested gives a smaller p-value than that study's own moderated t-test.

(meta-analysis-defaults)=
## Defaults and choices

- **Fixed or random effects.** The random-effects model by default, which lets the true effect vary between studies, so that a change in one study alone raises $\tau^2$ and widens the interval. The fixed-effect model (`method="fixed"`) answers for the studies at hand and assumes that they estimate one effect; it gives the most precise studies more weight than the random-effects model does.
- **The estimator of $\tau^2$.** REML, metafor's default; DerSimonian and Laird's, which needs no iterations; or Paule and Mandel's.
- **The test.** z-tests by default; the Knapp-Hartung adjustment with few studies, as metafor's documentation recommends.
- **Genes that some studies did not test.** `min_studies` (2) sets how many studies must test a gene.
- **Intervals.** `confidence_level` (0.95) sets the level of the intervals.

(meta-analysis-limits)=
## Limits and differences

- **Independent studies with known variances.** The studies are taken as independent, and their sampling variances as known. Meta-regression on study-level covariates, multivariate meta-analysis of correlated effects, and shrinkage across conditions such as mash [@urbut2019] are not implemented ([Not implemented yet](#not-implemented-yet)).
- **The largest maximum.** CytePool's REML estimate is the largest maximum of the restricted likelihood. metafor's Fisher scoring starts from the Hedges estimate, $\max(0, \sum_i (y_i - \bar y)^2 / (k - 1) - \bar v)$, and can stop at the smaller of two maxima, or set $\tau^2$ to zero, with a warning that Fisher scoring "may have gotten stuck at a local maximum", where the likelihood is higher at zero than at the maximum it found. Where the observed information exceeds twice the expected one, Fisher scoring oscillates: metafor then stops with an error, or at its default threshold somewhere in the oscillation, and the remedy of its help page, `control = list(stepadj = 0.5)`, converges to CytePool's estimate.
- **Estimates to rounding.** The REML and Paule-Mandel estimates converge to rounding. With its default settings metafor stops earlier, and where the restricted likelihood is largest at zero although the Hedges estimate is positive, its step halving approaches zero and stops near its threshold, short of zero.
- **No search interval.** metafor's Paule-Mandel search stops with an error when the root lies beyond $\max(100, 10\,\mathrm{mad}(y)^2)$, the end of its search interval; CytePool's search has no upper limit.
- **Standard errors.** A standard error must be positive and finite where it is given. For a zero standard error metafor stops, or fits by REML without $Q$, $I^2$ and $H^2$; for an infinite one it stops.

(meta-analysis-checks)=
## Checks and further reading

{api}`fit_meta_analysis` is compared with metafor 5.0.1's `rma.uni`, feature by feature, in ten cases from 2 to 30 studies, and {api}`meta_analysis` with metafor and R's `p.adjust` on the limma-voom tables of three simulated studies: exact (1e-10) for the fixed effect and DerSimonian-Laird, and close (1e-6) for REML and Paule-Mandel, run in metafor to convergence ([Validation](#validation-meta-analysis)).

Further reading:

- @viechtbauer2010 described metafor, the R package for meta-analysis that CytePool's meta-analysis follows.
- @dersimonian1986 introduced the random-effects meta-analysis of clinical trials with its moment estimator of the between-study variance.
- @higgins2002 introduced $I^2$ and $H^2$ to quantify heterogeneity.
- @viechtbauer2005 compared the bias and efficiency of the estimators of the between-study variance.

[Replication](replication.md), the last chapter, asks in how many studies a gene changes.
