---
description: Which CytePool function answers which question, for which design, and why the sample, not the cell, is the unit of every test.
---

# Choosing a method

CytePool's tests compare samples, not cells: each takes one profile per sample, so that
a difference is measured against the variation between donors. Cells of one donor share
its genetics and environment, and a test that treats them as independent replicates
finds differences that do not hold across donors [@squair2021; @zimmerman2021;
@murphy2022]. Pool the cells of each sample and cell type into a cube first
({api}`from_cells`); the methods below then run on its pseudobulk profiles.

## Genes that change in each cell type

| The design | Function | What it fits |
| --- | --- | --- |
| Conditions, with covariates or a pairing as fixed effects, such as `~ donor + condition` | {api}`limma_voom` | Log-CPM values with voom's precision weights, a weighted linear model, empirical Bayes moderation [@law2014; @ritchie2015] |
| The same, with library sizes within about threefold of each other | {api}`limma_trend` | Log-CPM values, a linear model, a prior variance that follows the expression level |
| Several samples of each donor, and a variable that differs between donors, such as disease or sex | {api}`dream_voom` | voom's weights from mixed models, a random intercept per donor, Satterthwaite's t-tests, empirical Bayes [@hoffman2021] |

Each runs on every group (cell type) of a cube, one at a time, and returns a
{api}`DifferentialExpressionResult` with a table for every cell type and contrast
({api}`DifferentialExpressionResult.table`), a summary, and which samples each fit used
and why the others were left out. The [limma User's
Guide](https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf)
(sections 15.4 and 15.5) says that limma-trend usually works well when the largest
library size is not more than about three times the smallest, and that voom is more
powerful when library sizes are quite variable. Pseudobulk library sizes grow with the
number of cells pooled, which can differ severalfold between samples, and
{api}`limma_voom` is the default.

A fixed effect per donor (`~ donor + condition`) compares the samples of each donor with
each other, which removes the differences between donors from the noise. It cannot test
a variable that is constant within each donor, such as sex: that variable is confounded
with the donor effects, and {api}`Design.check` says so. A random intercept per donor
({api}`dream_voom`) keeps the correlation between the samples of a donor and still
estimates donor-level effects. [Mixed models, REML and dream](mixed_models.md) explains
the model.

## One matrix of counts

Bulk RNA-seq, or one cell type taken out of a cube, is a samples x genes matrix of
counts. Three families of methods test it:

- **limma**: {api}`voom`, {api}`fit_linear_model`, {api}`contrasts_fit`, then
  {api}`moderated_statistics` or {api}`treat` for a fold-change threshold. A bulk
  experiment can also become a cube with one group ({api}`from_bulk`), on which
  {api}`limma_voom` runs the same steps.
- **edgeR's quasi-likelihood F-tests**: {api}`fit_quasi_likelihood` and
  {api}`quasi_likelihood_f_test`, negative binomial models with a moderated
  quasi-dispersion for every gene [@chen2025].
- **DESeq2**: {api}`fit_deseq2`, with {api}`deseq2_wald_test` for one coefficient or
  contrast and {api}`deseq2_likelihood_ratio_test` for several coefficients at once,
  such as every level of a factor [@love2014].

All three moderate the variability of each gene by borrowing from the others, and they
agree on genes with clear effects. They differ in the scale they model (log counts per
million, or counts) and in how they shrink: limma moderates residual variances towards a
prior, edgeR moderates quasi-dispersions, DESeq2 shrinks dispersions towards a trend in
the mean count. [Count models](count_models.md) gives the equations. In this version,
edgeR's and DESeq2's methods run on one matrix at a time, not on every group of a cube.

## Other questions

| Question | Functions | Explained in |
| --- | --- | --- |
| Do the genes of a pathway respond together? | {api}`camera`, {api}`fry`, {api}`roast`, {api}`mroast`, {api}`romer` | [Gene-set tests](gene_set_tests.md) |
| Does the share of a cell type change? | {api}`propeller` | [Composition tests](composition.md) |
| Do studies agree, and what is their common effect? | {api}`differential_expression_meta_analysis`, {api}`meta_analysis` | [Meta-analysis](meta_analysis.md) |
| How many of the calls are false? | {api}`adjust_p_values` | [Multiple testing](multiple_testing.md) |

## Choices that matter in every method

- **Samples with few cells.** A profile pooled from a handful of cells is noisy.
  `min_cells` (10 by default) leaves such samples out of a cell type's fit, and
  `min_counts` sets the least total count a sample needs; the scverse best practices
  keep samples with at least 10 cells and 1,000 counts [@heumos2023].
  `prior_weights="n_cells"` instead keeps every sample and weights it by its number of
  cells [@hoffman2026a].
- **Genes with low counts.** Genes are filtered by expression with the design, by
  edgeR's rule ({api}`filter_by_expression`; @chen2016), before voom's trend is fitted.
- **Outlying samples.** `quality_weights=True` estimates a weight for every sample from
  how well it follows the model, instead of removing it [@liu2015], and `robust=True`
  protects the prior from genes with outlying variances [@phipson2016].
- **Many cell types.** p-values are adjusted within each cell type and contrast;
  `adjust_across_groups=True` adds an adjustment over every cell type ([Multiple
  testing](multiple_testing.md)).
