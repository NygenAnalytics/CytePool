---
description: The terms the CytePool documentation uses, each in one sentence, with the section that explains it.
---

# Glossary

**Activity.** How strongly a transcription factor or a pathway acts in a sample or a contrast, read from the signed weights of a network on its targets: high where the genes it raises are high and those it lowers are low. [Networks of regulators and pathways](../statistics/activities.md#regulator-networks)

**Adjusted p-value.** A p-value corrected for the number of tests in its family, by default by the method of Benjamini and Hochberg, which controls the false discovery rate. [Error rates and their methods](../statistics/multiple_testing.md#error-rates)

**AICc.** Akaike's information criterion corrected for small samples, -2 log-likelihood + 2k + 2k(k + 1) / (n - k - 1) for k parameters and n samples, lower for a better balance of fit and size. [Read the curves table](../statistics/dose_response.md#dose-curve-table)

**Akaike weight.** A model's share of the support among the models fitted to a gene, from the differences of their AICc. [Read the curves table](../statistics/dose_response.md#dose-curve-table)

**Aliased column.** A column of a design matrix that is a combination of the columns before it, so that its coefficient cannot be estimated apart from theirs. [Designs that cannot answer the question](../statistics/linear_models.md#estimable-contrasts)

**B-statistic.** The log-odds that a gene's coefficient or contrast is not zero. [The B-statistic](../statistics/empirical_bayes.md#b-statistic)

**Benchmark dose (BMD).** The lowest dose at which a dose-response curve departs from the vehicle by a stated change, the benchmark response (BMR), such as one SD of the vehicle or a fold change of 1.1. [Simulated dose series with known curves](../statistics/dose_response.md#simulated-dose-series)

**Between-study variance (τ²).** The variance of the studies' true effects around their mean in a random-effects meta-analysis, zero in the fixed-effect model. [The between-study variance](../statistics/meta_analysis.md#between-study-variance)

**Candidate shapes.** The guesses of a dose response that the MCP step of MCP-Mod tests, each a curve of the change from the vehicle along dose in a model family such as Emax or the beta model's umbrella shape. [Trends, doses against the vehicle and candidate shapes](../statistics/dose_response.md#dose-response-contrasts)

**Competitive and self-contained tests.** A competitive gene-set test asks whether a set's genes change more than the other genes, and a self-contained test whether they change at all. [Competitive or self-contained](../statistics/gene_set_tests.md#competitive-and-self-contained)

**Confidence band (pointwise).** Intervals around a fitted curve, one at each dose, each of which holds with the stated probability, such as 95%, the curve that the model fits to the true means; that is the true curve where the response has the model's shape, and over the whole range at once the curve stays inside less often. [Curves with a band](../statistics/dose_response.md#dose-response-curves)

**Contrast.** A combination of a design's coefficients that answers one question, such as stimulated minus control. [Contrasts](../statistics/linear_models.md#contrasts)

**Contribution weight.** A weight that roast gives a gene of a set, whose sign says in which direction the gene is expected to change and whose size how much it counts. [roast and mroast](../statistics/gene_set_tests.md#roast)

**Cook's distance.** How much one sample's count moves a gene's fit, by which DESeq2 removes the p-value of a gene that one count drives. [DESeq2](../statistics/count_models.md#deseq2)

**Counts per million (CPM).** Counts divided by the effective library size, times one million. [Counts per million and log-CPM](../statistics/normalization_and_filtering.md#log-cpm)

**Coverage.** The share of intervals that contain the true value over repeated experiments, which a 95% interval should hold at 95%. [Simulated dose series with known curves](../statistics/dose_response.md#simulated-dose-series)

**Cox-Reid adjusted profile likelihood.** The likelihood of a gene's dispersion with the coefficients fitted at each value, adjusted for most of the bias that fitting them gives, which edgeR and DESeq2 maximize to estimate dispersions. [The Cox-Reid adjusted profile likelihood](../statistics/count_models.md#cox-reid)

**Cube.** CytePool's container of pseudobulk profiles: for every group, sample and feature, sums over cells, with the tables, fingerprint and provenance record that trace a result to its data. [What a cube records](../statistics/cubes.md#what-a-cube-records)

**Design.** The sample table with the model to fit: an R formula of the fixed effects, random effects in lme4's notation, and named contrasts. [The design](../statistics/linear_models.md#designs-and-formulas)

**Descriptive.** The status of a cell type whose dose series supports curves that describe the means of the dose groups, without p-values or bounds: no treated dose group that the design can compare with the vehicle has a second replicate (a second sample, or with random intercepts a second animal or other unit they name), the model leaves no residual degrees of freedom, or a batch is confounded with such a group and the design does not compare them through a random intercept of the batch. A cell type whose samples support tests is inferential, and one that supports no dose analysis, refused. [Checking a dose design](../statistics/dose_response.md#dose-design-check)

**Digest.** The SHA-256 hash of one part of a cube, such as a table or one group of a layer, from which the fingerprint is built. [Fingerprints](cube_files.md#fingerprint-encoding)

**Dispersion.** The parameter of a negative binomial distribution that sets how much the variance exceeds the mean, whose square root is the biological coefficient of variation. [The negative binomial model](../statistics/count_models.md#negative-binomial-model)

**Dispersion outlier.** A gene whose own dispersion estimate lies so far above DESeq2's trend that DESeq2 keeps it rather than shrink it towards the trend. [DESeq2](../statistics/count_models.md#deseq2)

**Dose axis.** The axis along which a curve of dose is fitted, usually log10 of the dose, with the unit, what the dose measures (administered, nominal, measured or internal), and where the vehicle goes: in a group of its own, or on the axis at a recorded offset. [Dose axes and the vehicle](../statistics/dose_response.md#dose-axes)

**Dose group.** The samples given one dose, the vehicle's included; in a cell type, the samples of a dose group that pass the cell type's filter of samples. [What a number of dose groups supports](../statistics/dose_response.md#dose-groups)

**Dunnett's test.** The comparison of each treatment or dose with the control at one family-wise error rate, which refers the largest of the contrasts' correlated t-statistics to the multivariate t distribution. [Dunnett's and Williams-type contrasts](../statistics/multiple_testing.md#dunnett-and-williams-contrasts)

**EC50.** The dose that gives half of a curve's maximal change. [Simulated dose series with known curves](../statistics/dose_response.md#simulated-dose-series)

**Effective library size.** A library size times its normalization factor, by which counts per million and the offsets of count models scale the counts. [Normalization factors and size factors](../statistics/normalization_and_filtering.md#normalization-and-size-factors)

**Empirical Bayes moderation.** Shrinking each gene's residual variance towards a value estimated from all the genes, which steadies the tests of experiments with few samples. [The model](../statistics/empirical_bayes.md#empirical-bayes-model)

**Estimable contrast.** A contrast that the samples determine, a combination of the rows of the design matrix, as every contrast of a design of full rank is. [Designs that cannot answer the question](../statistics/linear_models.md#estimable-contrasts)

**False discovery rate.** The expected share of false calls among all calls. [Error rates and their methods](../statistics/multiple_testing.md#error-rates)

**Family-wise error rate.** The chance of at least one false call among the tests of a family. [Error rates and their methods](../statistics/multiple_testing.md#error-rates)

**Feature.** A measured variable, usually a gene: a column of a samples x features matrix.

**Fingerprint.** A SHA-256 digest of what an analysis reads of a cube, its tables, cell numbers, normalization and layers, which names the cube however it was made. [What a cube records](../statistics/cubes.md#what-a-cube-records)

**Gate.** The test that decides which genes of a dose series count as responding: one test of any response at a stated false discovery rate, run after a filter of the genes that does not read the doses, with any fold change inside its null hypothesis. It takes the place of BMDExpress's prefilter, the step that picks the genes that go on to curves: BMDExpress's default prefilter takes an unadjusted Williams p-value and cuts the estimated fold change, where the gate adjusts its p-values and puts any fold change inside the test. The ANOVA gate of the National Toxicology Program tests the data set as a whole. [Which genes count as responding](../statistics/dose_response.md#dose-response-gate)

**Gene set.** A named list of genes that share a function or a response, such as a pathway or a signature, often read from a GMT file of a collection such as MSigDB. [Gene-set collections](../statistics/gene_set_tests.md#gene-set-collections)

**Gene set enrichment analysis (GSEA).** A test of whether the genes of a set gather at the top or the bottom of a ranking of all genes by a statistic, scored by the largest deviation of a walk down the ranking. [Gene set enrichment analysis](../statistics/over_representation_and_gsea.md#gsea)

**Group.** The first axis of a cube, usually a cell type, whose cells are pooled per sample, so that one group's layers form a samples x features matrix; a bulk cube has one group. [The layers](../statistics/cubes.md#cube-layers)

**Heterogeneity.** How much the studies of a meta-analysis disagree beyond their sampling error, measured by Cochran's Q, I² and the between-study variance. [Tests and heterogeneity](../statistics/meta_analysis.md#heterogeneity)

**I².** The share of the variation between the studies of a meta-analysis that comes from heterogeneity rather than from sampling error, reported as a proportion. [Tests and heterogeneity](../statistics/meta_analysis.md#heterogeneity)

**Independent filtering.** DESeq2's removal of the genes of low mean count before it adjusts the p-values, at the threshold that gives close to the most calls. [DESeq2](../statistics/count_models.md#deseq2)

**Inter-gene correlation.** The correlation between the genes of a set from sample to sample, for which camera inflates the variance of the set's mean statistic. [camera](../statistics/gene_set_tests.md#camera)

**Kenward-Roger's method.** An adjustment of the covariance of a mixed model's estimates, and of the degrees of freedom of their tests, for the uncertainty of the variance components with few samples. [Kenward-Roger's method](../statistics/mixed_models.md#kenward-roger)

**Knot.** A value at which the cubic pieces of a spline join; a natural cubic spline is a straight line beyond its two outer knots, the boundary knots, and a fitted curve cannot be reproduced without its knots. [Spline columns](../statistics/linear_models.md#spline-bases)

**Lack-of-fit test.** A test of a fitted curve against free means of the dose groups, whose small p-value says the curve misses the means, read at 0.1 for benchmark doses: an F-test in CytePool, and a likelihood ratio test, test 4, in BMDS. [Read the curves table](../statistics/dose_response.md#dose-curve-table)

**Layer.** One statistic of a cube summed over cells: the counts, the cells that detect each feature, or the sums of normalized and log-normalized values. [The layers](../statistics/cubes.md#cube-layers)

**Leading edge.** The genes of a set that drive its GSEA score, from the top of the ranking to the walk's extreme for a positive score, or from the extreme to the bottom for a negative one. [Gene set enrichment analysis](../statistics/over_representation_and_gsea.md#gsea)

**Leverage.** A sample's diagonal element of the hat matrix of a design, which says how much the fit at that sample rests on the sample itself, so that a leverage of 1 means a sample the model fits exactly. [Design checks](../statistics/linear_models.md#design-checks)

**Library size.** The total count of a cell, or of a sample or profile summed over its genes. [Library sizes and composition](../statistics/normalization_and_filtering.md#library-sizes)

**Linear trend test.** The moderated t-test of the slope of a straight line fitted to the dose groups' means along the dose axis, whether expression rises or falls steadily with log dose; not the Hill slope of a fitted curve. [Which genes respond to dose](../statistics/dose_response.md#dose-response-tests)

**Log-CPM.** The log2 of counts per million after a small count is added to every count, so that zero counts have finite logs. [Counts per million and log-CPM](../statistics/normalization_and_filtering.md#log-cpm)

**LOWESS span.** The share of the genes that each local fit of a LOWESS trend uses, which voom chooses from the number of genes. [The span](../statistics/limma_voom.md#voom-span)

**Lowest effective dose.** The lowest dose whose difference from the vehicle is called, in a gene that the gate passes, with no threshold on the size of the change: toxicology's lowest observed effect level (LOEL); the clinical minimum effective dose usually adds a clinically relevant size. [From which dose a gene responds](../statistics/dose_response.md#lowest-effective-dose)

**MCP step.** The multiple comparison procedure of MCP-Mod (multiple comparison procedure and modelling), its test step: each candidate shape becomes its optimal contrast of the dose groups, and the largest of their t-statistics is referred to the multivariate t distribution, which says whether the response follows any of the shapes. [Trends, doses against the vehicle and candidate shapes](../statistics/dose_response.md#dose-response-contrasts)

**Moderated F-statistic.** A test that several coefficients or contrasts of a gene are all zero at once, from their moderated t-statistics and the correlation of their estimates, on the number of independent contrasts and the gene's residual plus prior degrees of freedom. With dose groups alone in the design, it is a one-way ANOVA with moderated variances. [Moderated F-statistics](../statistics/empirical_bayes.md#moderated-f-statistics)

**Moderated t-statistic.** A coefficient or contrast divided by a standard error that uses the gene's moderated variance, tested on the gene's residual degrees of freedom plus the prior's. [The model](../statistics/empirical_bayes.md#empirical-bayes-model)

**Multidimensional scaling (MDS) plot.** limma's plot of the samples, placed so that their distances approximate the leading log2 fold changes between them. [Samples at a glance](../statistics/normalization_and_filtering.md#samples-at-a-glance)

**Multiple contrast test.** A test of several contrasts of one gene at once, such as each dose against the vehicle, whose p-values are adjusted for the gene's contrasts by the distribution of the largest of their correlated t-statistics. [Several contrasts of one gene](../statistics/multiple_testing.md#correlated-tests)

**Multivariate t distribution.** The distribution of correlated t-statistics that share their degrees of freedom: normal variables with a correlation matrix, divided by one chi variable. [Multivariate t probabilities](../statistics/multiple_testing.md#multivariate-t-probabilities)

**Natural cubic spline (restricted cubic spline).** A curve made of cubic pieces joined smoothly at knots and straight beyond the outer knots, which enters a linear model as a few columns of the design, in R's B-spline form or Harrell's restricted cubic form. [Spline columns](../statistics/linear_models.md#spline-bases)

**Network.** A map from each source, a transcription factor or a pathway, to its target genes with signed weights, such as CollecTRI's regulons or PROGENy's footprints. [Networks of regulators and pathways](../statistics/activities.md#regulator-networks)

**Non-linearity test.** The F-test of whether a fitted spline of dose bends away from a straight line along the dose axis: of its restricted cubic columns after the first, whose coefficients are all zero for a line. [Which genes respond to dose](../statistics/dose_response.md#dose-response-tests)

**Non-monotone response.** A response that rises and then falls along dose, or falls and then rises, such as a bell; one outlying dose group may instead drive a fitted curve that turns. [Find responses that turn](../statistics/dose_response.md#dose-curve-turns)

**Normalization factor and size factor.** Corrections for sequencing depth and composition: edgeR multiplies library sizes by normalization factors, such as TMM's, and DESeq2 replaces them by size factors. [Normalization factors and size factors](../statistics/normalization_and_filtering.md#normalization-and-size-factors)

**Normalized enrichment score.** A GSEA score divided by the mean of the scores of random sets of its sign, so that sets of different sizes compare. [p-values from random sets](../statistics/over_representation_and_gsea.md#gsea-p-values)

**One-sided test.** A test of a change in one stated direction, such as a fall with dose, which spends its whole error rate on that direction; for one t-statistic its p-value is half the two-sided one when t points the stated way, and 1 minus half of it otherwise. [Tests in one direction](../statistics/dose_response.md#one-sided-dose-tests)

**Optimal contrast.** The contrast of group means that best detects an expected shape, the one whose t-statistic has the largest non-centrality: for independent groups, each group's weight is proportional to its samples times the difference between the shape's mean in the group and the shape's mean weighted by the groups' samples. [Trends, doses against the vehicle and candidate shapes](../statistics/dose_response.md#dose-response-contrasts)

**Overall false discovery rate (OFDR).** The expected share of the genes called with at least one false call, the screen's or one of their contrasts', which stage-wise testing controls. [Calling contrasts across genes](../statistics/multiple_testing.md#stage-wise-testing)

**Over-representation analysis.** A test of whether the genes called significant hold more of a set's genes than chance would give, by the hypergeometric distribution over a universe of genes. [Over-representation analysis](../statistics/over_representation_and_gsea.md#over-representation)

**Partial conjunction.** The hypothesis that fewer than u of the studies that tested a gene have an effect, whose rejection claims an effect in at least u studies. [An effect in at least u studies](../statistics/replication.md#partial-conjunction)

**Pathway score.** One number per sample for a gene set, such as a GSVA score, that summarizes the expression of the set's genes in that sample, so that pathways can be compared across samples and tested like genes. [GSVA](../statistics/pathway_scores.md#gsva)

**Permutation check.** A check of whether the true doses of a dose series pass more genes through its gate than permuted doses do: the dose labels are permuted among the samples that the design treats alike, and the permutation p-value compares the genes the true doses pass with those of each permuted labelling, beside how often the gate passes genes with the doses permuted. [Could the calls come from noise?](../statistics/dose_response.md#dose-permutation-check)

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

**Rule set.** A named set of published rules that an analysis applies as one option, such as `rules="etap_2024"`, the design minimums of EPA's Transcriptomic Assessment Products. [ETAP's design minimums](../statistics/dose_response.md#dose-rule-minimums)

**Sample.** The unit of replication, such as a donor or a donor under one condition, whose cells are pooled into one profile per group.

**Sample quality weight.** A weight for each sample, from how closely its values follow the model across all genes. [Sample quality weights](../statistics/limma_voom.md#sample-quality-weights)

**Satterthwaite's degrees of freedom.** An approximation of the degrees of freedom of a mixed model's t-test, from how precisely the estimated variance components determine its standard error. [Satterthwaite's degrees of freedom](../statistics/mixed_models.md#satterthwaite)

**Scaled residual.** The difference of a dose group's mean from a fitted curve divided by the mean's standard error; a value above 2 in absolute value warrants a look. [Read the curves table](../statistics/dose_response.md#dose-curve-table)

**Single-step adjusted p-value.** The chance that the largest of a family's correlated test statistics is at least as extreme as a given one. It controls the family-wise error rate of the family, as Bonferroni's adjustment does, and takes the correlation into account. [Several contrasts of one gene](../statistics/multiple_testing.md#correlated-tests)

**Source record.** The caller's record of where a cube's data came from, kept in its provenance as it was given. [Provenance records](cube_files.md#provenance-keys)

**Spare dose groups.** The dose groups beyond a curve's parameters, K - p for K groups and p parameters; a curve needs one to test its fit, so a model of p parameters needs p + 1 dose groups. [The models of each set](../statistics/dose_response.md#dose-curve-sets)

**Stage-wise testing.** Calling the genes first, by a screen of any change at a false discovery rate, then their contrasts, each gene's adjusted for its own contrasts at the level the screen leaves. [Calling contrasts across genes](../statistics/multiple_testing.md#stage-wise-testing)

**SumRank.** A test of replication that ranks the genes within each study and sums every gene's ranks across the studies, so that a gene must rank high in study after study. [SumRank](../statistics/replication.md#sum-of-ranks)

**TREAT.** A test that a fold change exceeds a threshold, rather than that it is not zero. [Testing against a fold-change threshold](../statistics/empirical_bayes.md#treat-test)

**Umbrella shape.** A dose response that rises to a peak and falls, or falls to a trough and rises, which the beta model of the candidate shapes describes. [Trends, doses against the vehicle and candidate shapes](../statistics/dose_response.md#dose-response-contrasts)

**Universe.** The genes that an over-representation analysis could have selected, by default the genes tested. [The universe](../statistics/over_representation_and_gsea.md#over-representation-universe)

**Variance fraction.** The share of a gene's variance that one variable of the design explains, from a mixed model fitted by maximum likelihood, which variancePartition's help page reports gives unbiased fractions where REML can show substantial bias. [Variance partitioning](../statistics/mixed_models.md#variance-fractions)

**Vehicle.** The samples given the solvent or carrier of a treatment without the substance, dose 0 of a dose series, against which the doses are compared. [Dose axes and the vehicle](../statistics/dose_response.md#dose-axes)

**Williams-type contrasts.** Contrasts of the vehicle with the mean of the highest one, two or more doses, weighted by their group sizes, for a response that grows in dose order; unlike Williams' trend test, they need no fit of means that change in dose order. [Dunnett's and Williams-type contrasts](../statistics/multiple_testing.md#dunnett-and-williams-contrasts)
