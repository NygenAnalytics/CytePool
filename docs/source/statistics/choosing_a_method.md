---
description: Which CytePool function answers each question about samples, the default and when to choose another, and how the chapters that explain the methods follow from one another.
---

# Choosing a method

Use the table below to pick a starting function, see its R or Python equivalent, and find when to switch to an alternative. Every method operates on sample-level profiles rather than individual cells ([Why pseudobulk](../why_pseudobulk.ipynb)), starting from a cube built with {api}`from_cells` or {api}`from_bulk`.

| Question | Start with | R and Python equivalents | Choose instead | Chapter and guide |
| --- | --- | --- | --- | --- |
| Which genes change in each cell type? | {api}`limma_voom` | limma `voom`, `lmFit`, `eBayes`, `topTable`; edgeR `filterByExpr`, `calcNormFactors` | {api}`limma_trend` when library sizes lie within threefold; {api}`dream_voom` for repeated measures | [limma-voom and limma-trend](limma_voom.md), [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) |
| Which genes change in bulk RNA-seq? | {api}`limma_voom` on {api}`from_bulk` | edgeR `glmQLFit`, `glmQLFTest`; DESeq2 `DESeq`, `results`; limma `treat` | {api}`fit_quasi_likelihood` + {api}`quasi_likelihood_f_test`, or {api}`fit_deseq2` + {api}`deseq2_wald_test` / {api}`deseq2_likelihood_ratio_test` | [edgeR and DESeq2](count_models.md), [Bulk RNA-seq](../guides/bulk_rna_seq.ipynb) |
| Which genes change, with several samples of each donor? | {api}`dream_voom` | variancePartition `dream`, `voomWithDreamWeights`; lme4 `lmer`, lmerTest, pbkrtest | {api}`limma_voom` with `~ donor + condition` for within-donor questions, or on {api}`Cube.collapse_samples` between few donors | [Mixed models and dream](mixed_models.md), [Repeated measures and donors](../guides/repeated_measures.ipynb) |
| What drives the variation between samples? | {api}`variance_partitioning` | variancePartition `fitExtractVarPartModel`; DESeq2 `plotPCA`; limma `plotMDS` | {api}`principal_components`, {api}`plot_pca` and {api}`plot_mds` to inspect samples first | [Mixed models and dream](mixed_models.md), [Normalization and filtering](normalization_and_filtering.md) |
| Which cell types expand or contract? | {api}`propeller` | speckle `propeller` | select one lineage with `Cube.select` to test proportions within that lineage; {api}`plot_composition` | [Composition tests](composition.md), [Cell-type composition](../guides/composition.ipynb) |
| Do the genes of a pathway respond to a contrast? | {api}`camera` | limma `camera`, `fry`, `roast`, `mroast`, `romer`, `kegga` | {api}`fry`, {api}`roast` or {api}`mroast` for self-contained tests; {api}`romer`; {api}`over_representation` for gene lists | [Gene-set tests](gene_set_tests.md), [Gene sets and pathways](../guides/gene_sets.ipynb) |
| Do the genes of a pathway gather along a ranking? | {api}`gsea` | fgsea `fgseaMultilevel`, `fgseaSimple` | {api}`camera` when the linear model is available; {api}`plot_enrichment`, {api}`plot_gsea_table` | [Over-representation and GSEA](over_representation_and_gsea.md), [Gene sets and pathways](../guides/gene_sets.ipynb) |
| How active is a pathway in each sample? | {api}`gsva` | GSVA `gsva`, `ssgseaParam`, `zscoreParam`, `plageParam` | {api}`ssgsea`, {api}`combined_z_scores`, {api}`plage`; test scores with {api}`limma` | [Pathway scores per sample](pathway_scores.md), [Pathway and regulator activity](../guides/pathway_activity.ipynb) |
| Which regulators and pathways are active? | {api}`ulm` | decoupler `mt.ulm`, `mt.mlm`, `mt.waggr`; CollecTRI, PROGENy | {api}`mlm` for overlapping targets; {api}`weighted_aggregate` | [Activities from networks](activities.md), [Pathway and regulator activity](../guides/pathway_activity.ipynb) |
| What do several studies say together? | {api}`meta_analysis` | metafor `rma.uni`; metap | {api}`combine_p_values` when only p-values are comparable; {api}`CubeCollection.stack` for small uniform studies | [Aligning studies](aligning_studies.md), [Meta-analysis](meta_analysis.md), [Align and combine studies](../guides/combine_studies.ipynb) |
| Which changes replicate across studies? | {api}`sumrank` | SumRank; qvalue `pi0est` | {api}`replication` and {api}`partial_conjunction`; {api}`held_out_prediction` | [Replication](replication.md), [Replication across studies](../guides/replication_across_studies.ipynb) |

## Which genes change in each cell type?

Use {api}`limma_voom` to test contrasts across every cell type in a cube using voom precision weights [@law2014]. Switch to:

- {api}`limma_trend` when library sizes vary by less than about threefold;
- {api}`dream_voom` when donors contribute multiple samples ([below](#several-samples-of-each-donor)).

See [limma-voom and limma-trend](limma_voom.md) for the methods, and [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) for a worked example.

## Which genes change in bulk RNA-seq?

Pass a bulk count matrix to {api}`from_bulk` and run {api}`limma_voom`. Switch to:

- edgeR's quasi-likelihood F-tests ({api}`fit_quasi_likelihood` and {api}`quasi_likelihood_f_test`) or DESeq2 ({api}`fit_deseq2` and {api}`deseq2_wald_test`) to fit negative binomial models directly to counts [@chen2025; @love2014];
- {api}`quasi_likelihood_f_test` or {api}`deseq2_likelihood_ratio_test` to test multiple coefficients at once (such as all levels of a factor or time course);
- the step-by-step limma functions ({api}`voom`, {api}`fit_linear_model`, {api}`contrasts_fit`, {api}`moderated_statistics`), or {api}`treat` to test against a minimum log-fold-change threshold [@mccarthy2009].

edgeR and DESeq2 operate on a single count matrix (or one cell type extracted from a cube). See [edgeR and DESeq2](count_models.md) and [Bulk RNA-seq](../guides/bulk_rna_seq.ipynb).

(several-samples-of-each-donor)=
## Which genes change, with several samples of each donor?

Use {api}`dream_voom` to fit a random intercept per donor, which lets you test both within-donor changes and between-donor variables such as disease or sex [@hoffman2021]. Switch to:

- {api}`limma_voom` with donor as a fixed effect (`~ donor + condition`) when all comparisons are within donor (such as stimulated versus control);
- {api}`limma_voom` on donor-collapsed profiles ({api}`Cube.collapse_samples`) when testing a between-donor variable with very few donors per group (where dream can inflate false positives at n = 3 per group).

See [Mixed models and dream](mixed_models.md) and [Repeated measures and donors](../guides/repeated_measures.ipynb).

## What drives the variation between samples?

Use {api}`variance_partitioning` to estimate the fraction of each gene's variance explained by each variable in the design [@hoffman2016]. Use {api}`principal_components` ({api}`plot_pca`, {api}`plot_mds`) to inspect sample separation first. See [Mixed models and dream](mixed_models.md), [Normalization and filtering](normalization_and_filtering.md), [Repeated measures and donors](../guides/repeated_measures.ipynb) and [Normalization, filtering and sample quality](../guides/normalization_and_quality.ipynb).

## Which cell types expand or contract?

Use {api}`propeller` ({api}`plot_composition`) to test whether cell-type proportions change between conditions [@phipson2022]. See [Composition tests](composition.md) and [Cell-type composition](../guides/composition.ipynb).

## Do the genes of a pathway respond to a contrast?

Use {api}`camera` to test whether a gene set's genes change more than the remaining genes, accounting for inter-gene correlation [@wu2012]. Switch to:

- {api}`roast`, {api}`mroast` or {api}`fry` for self-contained tests of whether a set's genes change at all (roast for one set, mroast for multiple sets sharing rotations, fry for a fast analytic approximation across many sets);
- {api}`romer` for a rank-based rotation test on {api}`limma_trend` results;
- {api}`over_representation` for an unranked list of significant genes.

See [Gene-set tests](gene_set_tests.md), [Over-representation and GSEA](over_representation_and_gsea.md) and [Gene sets and pathways](../guides/gene_sets.ipynb).

## Do the genes of a pathway gather at the top or bottom of a ranking?

Use {api}`gsea` (matching fgsea) to test whether a gene set's genes are enriched at the top or bottom of a ranked gene list [@subramanian2005; @korotkevich2021]. `gsea` takes a per-gene ranking statistic from any differential-expression method. See [Over-representation and GSEA](over_representation_and_gsea.md) and [Gene sets and pathways](../guides/gene_sets.ipynb).

## How active is a pathway in each sample?

Use {api}`gsva` to compute a per-sample activity score for every gene set [@haenzelmann2013]. Switch to {api}`ssgsea`, {api}`combined_z_scores` or {api}`plage`, then test the scores across conditions with {api}`limma`. See [Pathway scores per sample](pathway_scores.md) and [Pathway and regulator activity](../guides/pathway_activity.ipynb).

## Which regulators and pathways are active?

Use {api}`ulm` (matching decoupler) to score transcription factors or pathways in each sample or contrast from signed target weights [@badiaimompel2022], using networks from {api}`collectri` or {api}`progeny`. Switch to {api}`mlm` when regulators share many targets, or {api}`weighted_aggregate` for permutation-tested weighted sums or means. See [Activities from networks](activities.md) and [Pathway and regulator activity](../guides/pathway_activity.ipynb).

## What do several studies say together?

Use {api}`meta_analysis` to combine per-gene effect estimates across studies with a random-effects model and estimate between-study heterogeneity [@viechtbauer2010]. Switch to:

- {api}`combine_p_values` when studies only provide p-values rather than comparable effect sizes;
- a joint model on stacked studies ({api}`CubeCollection.stack`) when small studies share the same protocol and platform.

See [Aligning studies](aligning_studies.md), [Meta-analysis](meta_analysis.md), [Replication](replication.md#combined-p-values), [Align and combine studies](../guides/combine_studies.ipynb) and [Replication across studies](../guides/replication_across_studies.ipynb).

## Which changes replicate across studies?

Use {api}`sumrank` to rank genes within each study and test the sum of ranks across studies for up- and down-regulated genes [@nakatsuka2025]. Switch to:

- {api}`replication` for cross-study concordance and {api}`partial_conjunction` to test whether a gene changes in at least u studies;
- {api}`held_out_prediction` to test whether genes selected in training studies classify samples in a held-out study.

See [Replication](replication.md) and [Replication across studies](../guides/replication_across_studies.ipynb).

(not-implemented-yet)=
## Not implemented yet

Each chapter's *Limits and differences* section documents how CytePool differs from the reference packages:

- **Differential expression:** edgeR and DESeq2 run on one matrix at a time rather than all cell types of a cube; edgeR's `voomLmFit`, `glmLRT` and gene-wise `estimateDisp`; DESeq2's `lfcShrink`, outlier replacement, `glmGamPoi` and local dispersion trends, and gene-specific normalization factors; limma's `duplicateCorrelation` and block correlations in voom.
- **Mixed models and composition:** random slopes `(tissue + 0 | donor)`, generalized linear mixed models, sample quality weights in dream, and bootstrap intervals (`varPartConfInf`); precision-weighted mixed models of composition (crumblr, @hoffman2026b) and Bayesian count models (scCODA, @buttner2021; sccomp, @mangiola2023).
- **Gene sets and activities:** gene-set tests on {api}`dream_voom` fits or with block correlations; gene length or abundance trends in over-representation (`kegga`, `goana`); gene weights in fry and exact z-scores in roast; decoupler's VIPER (whose reference implementation is licensed for academic use only).
- **Several studies:** meta-regression, multivariate meta-analysis, and effect shrinkage across conditions (mash, @urbut2019); r-values (@heller2014) and additional metap methods; study weights and random-effect permutations in SumRank; automatic ontology roll-up across the Cell Ontology.

(how-the-chapters-follow)=
## How the chapters follow each other

The chapters after this page explain the methods in an order where each builds on the ones before:

- **Foundations** cover what every method shares: [Pseudobulk profiles and cubes](cubes.md), the data every test reads; [Designs, linear models and contrasts](linear_models.md), the model fitted to every gene; [Empirical Bayes moderation](empirical_bayes.md), which shrinks each gene's variance towards a value estimated from all the genes; and [Multiple testing](multiple_testing.md), which turns thousands of p-values into a list of calls.
- **Differential expression** applies them to counts: [Normalization and filtering](normalization_and_filtering.md) prepares the counts, [limma-voom and limma-trend](limma_voom.md) tests every cell type, [edgeR and DESeq2](count_models.md) model the counts themselves, and [Mixed models and dream](mixed_models.md) allow for several samples of each donor.
- [Composition tests](composition.md) ask whether the shares of the cell types change.
- **Gene sets and pathways** turn the results for genes into results for pathways and regulators: [Gene-set tests](gene_set_tests.md), [Over-representation and GSEA](over_representation_and_gsea.md), [Pathway scores per sample](pathway_scores.md) and [Activities from networks](activities.md).
- **Several studies** compare and combine the results of studies: [Aligning studies](aligning_studies.md), [Meta-analysis](meta_analysis.md) and [Replication](replication.md).

The chapters begin with [Pseudobulk profiles and cubes](cubes.md).
