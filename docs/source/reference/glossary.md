---
description: The terms the CytePool documentation uses, each in one sentence, with the section that explains it.
---

# Glossary

**Activity.** How strongly a transcription factor or a pathway acts in a sample or a contrast, read from the signed weights of a network on its targets: high where the genes it raises are high and those it lowers are low. [Networks of regulators and pathways](../statistics/activities.md#regulator-networks)

**Adjusted p-value.** A p-value corrected for the number of tests in its family, by default by the method of Benjamini and Hochberg, which controls the false discovery rate. [Error rates and their methods](../statistics/multiple_testing.md#error-rates)

**Aliased column.** A column of a design matrix that is a combination of the columns before it, so that its coefficient cannot be estimated apart from theirs. [Designs that cannot answer the question](../statistics/linear_models.md#estimable-contrasts)

**B-statistic.** The log-odds that a gene's coefficient or contrast is not zero. [The B-statistic](../statistics/empirical_bayes.md#b-statistic)

**Between-study variance (τ²).** The variance of the studies' true effects around their mean in a random-effects meta-analysis, zero in the fixed-effect model. [The between-study variance](../statistics/meta_analysis.md#between-study-variance)

**Competitive and self-contained tests.** A competitive gene-set test asks whether a set's genes change more than the other genes, and a self-contained test whether they change at all. [Competitive or self-contained](../statistics/gene_set_tests.md#competitive-and-self-contained)

**Contrast.** A combination of a design's coefficients that answers one question, such as stimulated minus control. [Contrasts](../statistics/linear_models.md#contrasts)

**Contribution weight.** A weight that roast gives a gene of a set, whose sign says in which direction the gene is expected to change and whose size how much it counts. [roast and mroast](../statistics/gene_set_tests.md#roast)

**Cook's distance.** How much one sample's count moves a gene's fit, by which DESeq2 removes the p-value of a gene that one count drives. [DESeq2](../statistics/count_models.md#deseq2)

**Counts per million (CPM).** Counts divided by the effective library size, times one million. [Counts per million and log-CPM](../statistics/normalization_and_filtering.md#log-cpm)

**Cox-Reid adjusted profile likelihood.** The likelihood of a gene's dispersion with the coefficients fitted at each value, adjusted for most of the bias that fitting them gives, which edgeR and DESeq2 maximize to estimate dispersions. [The Cox-Reid adjusted profile likelihood](../statistics/count_models.md#cox-reid)

**Cube.** CytePool's container of pseudobulk profiles: for every group, sample and feature, sums over cells, with the tables, fingerprint and provenance record that trace a result to its data. [What a cube records](../statistics/cubes.md#what-a-cube-records)

**Design.** The sample table with the model to fit: an R formula of the fixed effects, random effects in lme4's notation, and named contrasts. [The design](../statistics/linear_models.md#designs-and-formulas)

**Digest.** The SHA-256 hash of one part of a cube, such as a table or one group of a layer, from which the fingerprint is built. [Fingerprints](cube_files.md#fingerprint-encoding)

**Dispersion.** The parameter of a negative binomial distribution that sets how much the variance exceeds the mean, whose square root is the biological coefficient of variation. [The negative binomial model](../statistics/count_models.md#negative-binomial-model)

**Dispersion outlier.** A gene whose own dispersion estimate lies so far above DESeq2's trend that DESeq2 keeps it rather than shrink it towards the trend. [DESeq2](../statistics/count_models.md#deseq2)

**Effective library size.** A library size times its normalization factor, by which counts per million and the offsets of count models scale the counts. [Normalization factors and size factors](../statistics/normalization_and_filtering.md#normalization-and-size-factors)

**Empirical Bayes moderation.** Shrinking each gene's residual variance towards a value estimated from all the genes, which steadies the tests of experiments with few samples. [The model](../statistics/empirical_bayes.md#empirical-bayes-model)

**Estimable contrast.** A contrast that the samples determine, a combination of the rows of the design matrix, as every contrast of a design of full rank is. [Designs that cannot answer the question](../statistics/linear_models.md#estimable-contrasts)

**False discovery rate.** The expected share of false calls among all calls. [Error rates and their methods](../statistics/multiple_testing.md#error-rates)

**Family-wise error rate.** The chance of at least one false call among the tests of a family. [Error rates and their methods](../statistics/multiple_testing.md#error-rates)

**Feature.** A measured variable, usually a gene: a column of a samples x features matrix.

**Fingerprint.** A SHA-256 digest of what an analysis reads of a cube, its tables, cell numbers, normalization and layers, which names the cube however it was made. [What a cube records](../statistics/cubes.md#what-a-cube-records)

**Gene set.** A named list of genes that share a function or a response, such as a pathway or a signature, often read from a GMT file of a collection such as MSigDB. [Gene-set collections](../statistics/gene_set_tests.md#gene-set-collections)

**Gene set enrichment analysis (GSEA).** A test of whether the genes of a set gather at the top or the bottom of a ranking of all genes by a statistic, scored by the largest deviation of a walk down the ranking. [Gene set enrichment analysis](../statistics/over_representation_and_gsea.md#gsea)

**Group.** The first axis of a cube, usually a cell type, whose cells are pooled per sample, so that one group's layers form a samples x features matrix; a bulk cube has one group. [The layers](../statistics/cubes.md#cube-layers)

**Heterogeneity.** How much the studies of a meta-analysis disagree beyond their sampling error, measured by Cochran's Q, I² and the between-study variance. [Tests and heterogeneity](../statistics/meta_analysis.md#heterogeneity)

**I².** The share of the variation between the studies of a meta-analysis that comes from heterogeneity rather than from sampling error, reported as a proportion. [Tests and heterogeneity](../statistics/meta_analysis.md#heterogeneity)

**Independent filtering.** DESeq2's removal of the genes of low mean count before it adjusts the p-values, at the threshold that gives close to the most calls. [DESeq2](../statistics/count_models.md#deseq2)

**Inter-gene correlation.** The correlation between the genes of a set from sample to sample, for which camera inflates the variance of the set's mean statistic. [camera](../statistics/gene_set_tests.md#camera)

**Kenward-Roger's method.** An adjustment of the covariance of a mixed model's estimates, and of the degrees of freedom of their tests, for the uncertainty of the variance components with few samples. [Kenward-Roger's method](../statistics/mixed_models.md#kenward-roger)

**Layer.** One statistic of a cube summed over cells: the counts, the cells that detect each feature, or the sums of normalized and log-normalized values. [The layers](../statistics/cubes.md#cube-layers)

**Leading edge.** The genes of a set that drive its GSEA score, from the top of the ranking to the walk's extreme for a positive score, or from the extreme to the bottom for a negative one. [Gene set enrichment analysis](../statistics/over_representation_and_gsea.md#gsea)

**Leverage.** A sample's diagonal element of the hat matrix of a design, which says how much the fit at that sample rests on the sample itself, so that a leverage of 1 means a sample the model fits exactly. [Design checks](../statistics/linear_models.md#design-checks)

**Library size.** The total count of a cell, or of a sample or profile summed over its genes. [Library sizes and composition](../statistics/normalization_and_filtering.md#library-sizes)

**Log-CPM.** The log2 of counts per million after a small count is added to every count, so that zero counts have finite logs. [Counts per million and log-CPM](../statistics/normalization_and_filtering.md#log-cpm)

**LOWESS span.** The share of the genes that each local fit of a LOWESS trend uses, which voom chooses from the number of genes. [The span](../statistics/limma_voom.md#voom-span)

**Moderated t-statistic.** A coefficient or contrast divided by a standard error that uses the gene's moderated variance, tested on the gene's residual degrees of freedom plus the prior's. [The model](../statistics/empirical_bayes.md#empirical-bayes-model)

**Multidimensional scaling (MDS) plot.** limma's plot of the samples, placed so that their distances approximate the leading log2 fold changes between them. [Samples at a glance](../statistics/normalization_and_filtering.md#samples-at-a-glance)

**Network.** A map from each source, a transcription factor or a pathway, to its target genes with signed weights, such as CollecTRI's regulons or PROGENy's footprints. [Networks of regulators and pathways](../statistics/activities.md#regulator-networks)

**Normalization factor and size factor.** Corrections for sequencing depth and composition: edgeR multiplies library sizes by normalization factors, such as TMM's, and DESeq2 replaces them by size factors. [Normalization factors and size factors](../statistics/normalization_and_filtering.md#normalization-and-size-factors)

**Normalized enrichment score.** A GSEA score divided by the mean of the scores of random sets of its sign, so that sets of different sizes compare. [p-values from random sets](../statistics/over_representation_and_gsea.md#gsea-p-values)

**Over-representation analysis.** A test of whether the genes called significant hold more of a set's genes than chance would give, by the hypergeometric distribution over a universe of genes. [Over-representation analysis](../statistics/over_representation_and_gsea.md#over-representation)

**Partial conjunction.** The hypothesis that fewer than u of the studies that tested a gene have an effect, whose rejection claims an effect in at least u studies. [An effect in at least u studies](../statistics/replication.md#partial-conjunction)

**Pathway score.** One number per sample for a gene set, such as a GSVA score, that summarizes the expression of the set's genes in that sample, so that pathways can be compared across samples and tested like genes. [GSVA](../statistics/pathway_scores.md#gsva)

**Precision weight.** A weight for each observation, the inverse of the variance that a mean-variance trend predicts for it, such as voom's. [voom](../statistics/limma_voom.md#voom)

**Principal components.** The directions along which the samples' values vary most, from the singular value decomposition of the centred values of their most variable genes, on which DESeq2's PCA plot places the samples. [Samples at a glance](../statistics/normalization_and_filtering.md#samples-at-a-glance)

**Prior degrees of freedom.** How much the value that all genes share counts in empirical Bayes moderation, as a number of degrees of freedom added to each gene's own. [The model](../statistics/empirical_bayes.md#empirical-bayes-model)

**Proportion of true nulls (π₀).** The share of a family's tests without an effect, estimated from the p-values above a threshold, since those of tests without an effect are uniform. [The proportion of true nulls](../statistics/multiple_testing.md#proportion-of-true-nulls)

**Proportion with an effect (π₁).** One minus π₀, which, estimated from one study's p-values of the genes that another study finds, gives the share of those genes that change in the first study, found there or not. [How well one study replicates another](../statistics/replication.md#pi1-between-studies)

**Provenance record.** The record of a cube's source, of how it was built and of every operation since, each operation with the fingerprint of its input where that is known. [What a cube records](../statistics/cubes.md#what-a-cube-records)

**Pseudobulk profile.** The sums over the cells of one sample and group, which make the sample, not the cell, the unit of replication. [Why pseudobulk](../why_pseudobulk.ipynb)

**Quasi-dispersion.** In edgeR's quasi-likelihood model, a gene's own multiplier of the negative binomial variance, which empirical Bayes moderates. [edgeR's quasi-likelihood F-tests](../statistics/count_models.md#quasi-likelihood)

**Random intercept.** A shift shared by the samples of one level of a grouping factor, such as a donor, drawn from a normal distribution whose variance is estimated. [The model](../statistics/mixed_models.md#mixed-model)

**Reference level.** The first level of a factor, from which the coefficients of its other levels are differences in a design with an intercept. [The design](../statistics/linear_models.md#designs-and-formulas)

**REML.** Restricted maximum likelihood, which estimates variance components from the part of the data that the fixed effects do not fit, without the downward bias that estimating the fixed effects gives maximum likelihood estimates. [REML](../statistics/mixed_models.md#mixed-model-reml)

**Residual degrees of freedom.** What a fit leaves to estimate the variance: the number of samples it uses minus the number of coefficients, or, in a mixed model, an approximation between $n - p$ and $n$ minus the rank of the fixed and random effects together. [REML](../statistics/mixed_models.md#mixed-model-reml)

**Robust moderation.** Empirical Bayes moderation whose prior is estimated so that genes with outlying variances count less and keep more of their own variance. [Estimating the prior](../statistics/empirical_bayes.md#estimating-the-prior)

**Rotation test.** A test that compares a statistic with its values under random rotations of a linear model's residual space, which leave the distribution of the data, the correlation between genes included, unchanged under the null hypothesis. [The effects of each gene](../statistics/gene_set_tests.md#gene-effects)

**Sample.** The unit of replication, such as a donor or a donor under one condition, whose cells are pooled into one profile per group.

**Sample quality weight.** A weight for each sample, from how closely its values follow the model across all genes. [Sample quality weights](../statistics/limma_voom.md#sample-quality-weights)

**Satterthwaite's degrees of freedom.** An approximation of the degrees of freedom of a mixed model's t-test, from how precisely the estimated variance components determine its standard error. [Satterthwaite's degrees of freedom](../statistics/mixed_models.md#satterthwaite)

**Source record.** The caller's record of where a cube's data came from, kept in its provenance as it was given. [Provenance records](cube_files.md#provenance-keys)

**SumRank.** A test of replication that ranks the genes within each study and sums every gene's ranks across the studies, so that a gene must rank high in study after study. [SumRank](../statistics/replication.md#sum-of-ranks)

**TREAT.** A test that a fold change exceeds a threshold, rather than that it is not zero. [Testing against a fold-change threshold](../statistics/empirical_bayes.md#treat-test)

**Universe.** The genes that an over-representation analysis could have selected, by default the genes tested. [The universe](../statistics/over_representation_and_gsea.md#over-representation-universe)

**Variance fraction.** The share of a gene's variance that one variable of the design explains, from a mixed model fitted by maximum likelihood, which variancePartition's help page reports gives unbiased fractions where REML can show substantial bias. [Variance partitioning](../statistics/mixed_models.md#variance-fractions)
