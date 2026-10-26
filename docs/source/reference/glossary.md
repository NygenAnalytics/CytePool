---
description: The terms the CytePool documentation uses, each in one or two sentences, with the page that explains it.
---

# Glossary

**Adjusted p-value.** A p-value corrected for the number of tests, here usually by the
method of Benjamini and Hochberg, which controls the false discovery rate. [Multiple
testing](../statistics/multiple_testing.md)

**B-statistic.** The log-odds that a gene's coefficient or contrast is not zero.
[Empirical Bayes moderation](../statistics/empirical_bayes.md)

**Competitive and self-contained tests.** A competitive gene-set test asks whether a
set's genes change more than the other genes; a self-contained test asks whether they
change at all. [Gene-set tests](../statistics/gene_set_tests.md)

**Contrast.** A combination of a design's coefficients that answers one question, such
as stimulated minus control. It is *estimable* when the samples determine it. [Designs,
linear models and contrasts](../statistics/linear_models.md)

**Counts per million (CPM).** Counts divided by the effective library size, times one
million; log-CPM is their log2, after a small count is added. [Count
models](../statistics/count_models.md)

**Cube.** CytePool's container of pseudobulk profiles: for every sample, group and
feature, sums over cells, with tables that describe the samples, groups and features.
[The cube and its additive layers](../statistics/cubes.md)

**Design.** The sample table with the model to fit: an R formula for the fixed effects,
random effects in lme4's notation, and named contrasts. [Designs, linear models and
contrasts](../statistics/linear_models.md)

**Digest.** The SHA-256 hash of one part of a cube, such as a table or one group of a
layer, from which the fingerprint is built. [Cube files and provenance
records](cube_files.md)

**Dispersion.** The parameter of a negative binomial distribution that sets how much the
variance exceeds the mean; its square root is the biological coefficient of variation.
[Count models](../statistics/count_models.md)

**Empirical Bayes moderation.** Shrinking each gene's variance towards a value estimated
from all genes, which steadies tests with few samples. The prior degrees of freedom say
how much the shared value counts. [Empirical Bayes
moderation](../statistics/empirical_bayes.md)

**False discovery rate.** The expected share of false calls among all calls. [Multiple
testing](../statistics/multiple_testing.md)

**Feature.** A measured variable, usually a gene: a column of a samples x features
matrix.

**Fingerprint.** A digest of a cube's values, its tables, cell numbers and layers
included, that names the cube however it was made. [Cube files and provenance
records](cube_files.md)

**Gene set.** A named list of genes, such as a pathway or a signature, from a collection
such as MSigDB, often read from a GMT file. [Gene-set
tests](../statistics/gene_set_tests.md#gene-set-files)

**Gene set enrichment analysis (GSEA).** A test of whether the genes of a set gather at
the top or the bottom of a ranking of all genes by a statistic, from the largest
deviation of a walk down the ranking, its enrichment score. [Gene-set
tests](../statistics/gene_set_tests.md#gene-set-enrichment-analysis)

**Group.** The second axis of a cube, usually a cell type: the cells of one sample are
pooled per group. A bulk cube has one group.

**Heterogeneity.** How much the studies of a meta-analysis disagree beyond their
sampling error, measured by the between-study variance tau², Cochran's Q and I².
[Meta-analysis](../statistics/meta_analysis.md)

**Kenward-Roger's method.** An adjustment of the standard errors of a mixed model's
estimates, and of the degrees of freedom of their tests, for the uncertainty of its
variance components in small samples. [Mixed models, REML and dream](../statistics/mixed_models.md)

**Layer.** One statistic of a cube, summed over cells: counts, detected cells, or sums
of normalized and log-normalized values. [The cube and its additive
layers](../statistics/cubes.md)

**Library size.** The total count of a sample, or of a profile; times a normalization
factor, the effective library size.

**Normalization factor and size factor.** Corrections for sequencing depth and
composition: edgeR multiplies library sizes by normalization factors (such as TMM), and
DESeq2 replaces them by size factors. [Count models](../statistics/count_models.md)

**Over-representation analysis.** A test of whether the genes called significant hold
more of a set's genes than chance would give, by the hypergeometric distribution against
a universe of genes. [Gene-set
tests](../statistics/gene_set_tests.md#over-representation-analysis)

**Precision weight.** A weight for each observation, the inverse of its predicted
variance, such as voom's. [Precision weights](../statistics/precision_weights.md)

**Provenance record.** The record of where a cube's data came from (its source), how it
was built and every operation since, each naming its input by fingerprint. [Cube files
and provenance records](cube_files.md)

**Pseudobulk profile.** The sums over the cells of one sample and group, which treats
the sample, not the cell, as the unit of replication. [Choosing a
method](../statistics/choosing_a_method.md)

**Quasi-dispersion.** In edgeR's quasi-likelihood model, a gene's own multiplier of the
negative binomial variance. [Count models](../statistics/count_models.md)

**Random intercept.** A shift shared by the samples of one level of a grouping factor,
such as a donor, drawn from a normal distribution whose variance is estimated. [Mixed
models, REML and dream](../statistics/mixed_models.md)

**Reference level.** The first level of a factor, against which the coefficients of its
other levels are differences. [Designs, linear models and
contrasts](../statistics/linear_models.md)

**REML.** Restricted maximum likelihood, which estimates variance components without the
bias that estimating the fixed effects gives maximum likelihood. [Mixed models, REML and
dream](../statistics/mixed_models.md)

**Residual degrees of freedom.** The number of samples a fit uses minus the number of
coefficients it estimates: what is left to estimate the variance.

**Rotation test.** A test that compares a statistic with its values under random
rotations of a linear model's residual space, which leave the data's distribution
unchanged under the null hypothesis. [Gene-set tests](../statistics/gene_set_tests.md)

**Sample.** The unit of replication: a donor, or a donor under one condition. Its cells
are pooled into one profile per group.

**Sample quality weight.** A weight for each sample, from how closely its values follow
the model across genes. [Precision weights](../statistics/precision_weights.md)

**Satterthwaite's degrees of freedom.** An approximation to the degrees of freedom of a
t-test in a mixed model. [Mixed models, REML and dream](../statistics/mixed_models.md)

**Source record.** The caller's record of where a cube's data came from, kept in its
provenance as given. [Cube files and provenance records](cube_files.md)

**TREAT.** A test that a fold change exceeds a threshold, rather than that it is not
zero. [Empirical Bayes moderation](../statistics/empirical_bayes.md)
