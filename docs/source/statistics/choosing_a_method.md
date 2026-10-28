---
description: Which CytePool function answers each question about samples, the default and when to choose another, and how the chapters that explain the methods follow from one another.
---

# Choosing a method

A study that compares conditions, donors or treatments asks a few questions of its
samples: which genes change, which cell types expand, which pathways respond, and whether
a finding holds in other studies. This page lists each question with the function that
answers it by default, the alternatives and when to choose them, the chapter that
explains the method and the guide that runs it on data. Every test compares samples, not
cells, for the reasons [Why pseudobulk](../why_pseudobulk.ipynb) gives, so the cells of
each sample and cell type are first pooled into a cube ({api}`from_cells`).

(how-the-chapters-follow)=
## How the chapters follow each other

The chapters after this page explain the methods in an order where each builds on the
ones before:

- **Foundations** cover what every method shares: [Pseudobulk profiles and
  cubes](cubes.md), the data every test reads; [Designs, linear models and
  contrasts](linear_models.md), the model fitted to every gene; [Empirical Bayes
  moderation](empirical_bayes.md), which shrinks each gene's variance towards a value
  estimated from all the genes; and [Multiple testing](multiple_testing.md), which turns
  thousands of p-values into a list of calls.
- **Differential expression** applies them to counts: [Normalization and
  filtering](normalization_and_filtering.md) prepares the counts, [limma-voom and
  limma-trend](limma_voom.md) tests every cell type, [edgeR and DESeq2](count_models.md)
  model the counts themselves, and [Mixed models and dream](mixed_models.md) allow for
  several samples of each donor.
- [Composition tests](composition.md) ask whether the shares of the cell types change.
- **Gene sets and pathways** turn the results for genes into results for pathways and
  regulators: [Gene-set tests](gene_set_tests.md), [Over-representation and
  GSEA](over_representation_and_gsea.md), [Pathway scores per
  sample](pathway_scores.md) and [Activities from networks](activities.md).
- **Several studies** compare and combine the results of studies: [Aligning
  studies](aligning_studies.md), [Meta-analysis](meta_analysis.md) and
  [Replication](replication.md).

## Which genes change in each cell type?

{api}`limma_voom` tests the contrasts of a design in every cell type of a cube, with
voom's precision weights [@law2014]. Choose instead:

- {api}`limma_trend` when library sizes lie within about threefold of each other;
- {api}`dream_voom` when donors give several samples each ([below](#several-samples-of-each-donor)).

[limma-voom and limma-trend](limma_voom.md) explains both, and [Genes that change in
each cell type](../guides/genes_per_cell_type.ipynb) runs them on published data.

## Which genes change in bulk RNA-seq?

{api}`limma_voom` on a cube with one group, which {api}`from_bulk` makes from a matrix of
counts: the analysis of pseudobulk profiles, applied to bulk samples. Choose instead:

- edgeR's quasi-likelihood F-tests ({api}`fit_quasi_likelihood` and
  {api}`quasi_likelihood_f_test`) or DESeq2 ({api}`fit_deseq2` and
  {api}`deseq2_wald_test`), which model the counts themselves with a negative binomial
  distribution [@chen2025; @love2014];
- to test several coefficients together, all the levels of a factor or all the time
  points, the F-tests of {api}`quasi_likelihood_f_test` or DESeq2's likelihood ratio test
  ({api}`deseq2_likelihood_ratio_test`);
- the steps of limma on a matrix of counts, {api}`voom`, {api}`fit_linear_model`,
  {api}`contrasts_fit` and {api}`moderated_statistics`, or {api}`treat` to test against
  a fold-change threshold [@mccarthy2009].

edgeR's and DESeq2's methods run on one matrix of counts, such as one cell type taken
out of a cube. [edgeR and DESeq2](count_models.md) explains them, and [Bulk
RNA-seq](../guides/bulk_rna_seq.ipynb) runs the three methods on a simulated
experiment.

(several-samples-of-each-donor)=
## Which genes change, with several samples of each donor?

{api}`dream_voom` gives each donor a random intercept, so that it tests both changes
within donors and variables of the donors, such as disease or sex [@hoffman2021]. Choose
instead:

- {api}`limma_voom` with a fixed effect for each donor (`~ donor + condition`) when
  every question is within donors, such as stimulated against control in each donor;
- {api}`limma_voom` on the profiles pooled by donor ({api}`Cube.collapse_samples`), when
  every donor was sampled alike, to test a variable of the donors with few donors in
  each group, where dream's tests between donors made too many false calls with three
  donors per group.

[Mixed models and dream](mixed_models.md) explains the models, and [Repeated measures
and donors](../guides/repeated_measures.ipynb) compares them on data.

## What drives the variation between samples?

{api}`variance_partitioning` gives, in every cell type, how much of each gene's
variance each variable of the design explains [@hoffman2016]. To look at the samples
first, {api}`principal_components` places them on the principal components of the genes
that vary most. [Mixed models and dream](mixed_models.md) and [Normalization
and filtering](normalization_and_filtering.md) explain them, and [Repeated measures and
donors](../guides/repeated_measures.ipynb) and [Normalization, filtering and sample
quality](../guides/normalization_and_quality.ipynb) show them on data.

## Which cell types expand or contract?

{api}`propeller` tests whether the share of each cell type differs between conditions,
from the cell numbers of a cube [@phipson2022]. [Composition tests](composition.md)
explains it, and [Cell-type composition](../guides/composition.ipynb) runs it on a
simulated study.

## Do the genes of a pathway respond to a contrast?

{api}`camera` tests whether the genes of each set of a collection change more than the
other genes, and allows for the correlation between them [@wu2012]. Choose instead:

- {api}`roast`, {api}`mroast` or {api}`fry` for sets chosen in advance, to ask whether
  their genes change at all: roast for one set, mroast for several with the same
  rotations, and fry, which approximates their test of the mean without rotations, for
  any number of sets;
- {api}`romer`, which compares the ranks of a set's genes with those of the others, on a
  result of {api}`limma_trend`;
- {api}`over_representation` for a list of genes, such as those that another method
  called significant.

[Gene-set tests](gene_set_tests.md) and [Over-representation and
GSEA](over_representation_and_gsea.md) explain them, and [Gene sets and
pathways](../guides/gene_sets.ipynb) runs camera, fry, roast and over-representation
analysis on published data.

## Do the genes of a pathway gather at the top or bottom of a ranking?

{api}`gsea`, as fgsea computes it, tests whether a set's genes gather at either end of
the genes ordered by a statistic [@subramanian2005; @korotkevich2021]. It
needs only a statistic for every gene, from any method. [Over-representation and
GSEA](over_representation_and_gsea.md) explains it, and [Gene sets and
pathways](../guides/gene_sets.ipynb) runs it.

## How active is a pathway in each sample?

{api}`gsva` scores every gene set in every sample [@haenzelmann2013]. Choose instead
{api}`ssgsea`, which also compares a set's genes with the other genes of each sample, or
{api}`combined_z_scores` and {api}`plage`, which follow each set across the samples.
{api}`limma` tests the scores with the cube's design. [Pathway scores per
sample](pathway_scores.md) explains them, and [Pathway and regulator
activity](../guides/pathway_activity.ipynb) runs GSVA and tests its scores.

## Which regulators and pathways are active?

{api}`ulm` scores every transcription factor or pathway of a network, in every sample or
contrast, from the signs and sizes of its weights on its targets, as decoupler computes
it [@badiaimompel2022]; {api}`collectri` and {api}`progeny` give the networks. Choose
instead {api}`mlm` to separate sources whose targets overlap, or
{api}`weighted_aggregate` for the weighted mean or sum of the targets, with
permutations. [Activities from networks](activities.md) explains them, and [Pathway and
regulator activity](../guides/pathway_activity.ipynb) runs ULM.

## What do several studies say together?

{api}`meta_analysis` combines, for every gene, the estimates of several studies into one,
by a random-effects model, and measures how far the studies disagree [@viechtbauer2010].
Choose instead:

- {api}`combine_p_values` where the studies give p-values rather than comparable
  effects;
- a joint model of the stacked studies ({api}`CubeCollection.stack`) for studies made
  alike, with one protocol and platform, that are too small to estimate their variances
  alone.

[Aligning studies](aligning_studies.md) and [Meta-analysis](meta_analysis.md) explain
them, and [Replication](replication.md#combined-p-values) the combined p-values. [Align
and combine studies](../guides/combine_studies.ipynb) runs the meta-analysis
and the joint model on six simulated cohorts, and [Replication across
studies](../guides/replication_across_studies.ipynb) combines their p-values.

## Which changes replicate across studies?

{api}`sumrank` ranks every gene within each study and sums its ranks across the studies,
by default those of every study, for genes up and down [@nakatsuka2025]. Choose instead:

- {api}`replication` for the genes found in each number of studies, a test of a change
  in at least u studies ({api}`partial_conjunction`), and the agreement between pairs of
  studies;
- {api}`held_out_prediction` to ask whether the genes chosen from some studies predict
  the cases of another.

[Replication](replication.md) explains them, and [Replication across
studies](../guides/replication_across_studies.ipynb) runs them on six simulated cohorts.

(not-implemented-yet)=
## Not implemented yet

These methods have no CytePool function yet:

- **Genes that change.** edgeR's quasi-likelihood F-tests and DESeq2's tests on every
  cell type of a cube; edgeR's likelihood ratio tests and gene-wise dispersions
  (`glmLRT`, `estimateDisp`) and its `legacy = TRUE` methods; DESeq2's replacement of
  outlier counts, shrunken fold changes (`lfcShrink`), local and glmGamPoi dispersion
  trends, observation weights, normalization factors that differ between genes,
  fold-change thresholds (`lfcThreshold`) and t-distributed Wald tests (`useT`);
  limma's `duplicateCorrelation`, voom with a block correlation, voom's normalization
  of log-CPM values between samples (`normalize.method`), and edgeR's `voomLmFit`.
- **Several samples of each donor.** Random slopes in mixed models, and with them the
  partition of the variance within subsets of the samples; sample quality weights for
  dream; generalized linear mixed models; bootstrap intervals of the variance
  fractions (variancePartition's `varPartConfInf`).
- **Composition.** Precision-weighted mixed models of composition, as crumblr fits them
  [@hoffman2026b], and Bayesian count models such as scCODA [@buttner2021] and sccomp
  [@mangiola2023].
- **Gene sets.** Gene-set tests of the mixed models of {api}`dream_voom`, and with a
  block correlation (romer's `block`); limma's gene contribution weights in fry
  (`gene.weights`); the adjustment of over-representation for gene length or abundance
  (the `trend` of limma's `kegga` and `goana`); exact z-scores and given priors in the rotation tests
  (`approx.zscore = FALSE`, `var.prior`, `df.prior`). decoupler's VIPER is not offered,
  because the licence of its reference implementation allows only non-profit academic
  use.
- **Several studies.** Meta-regression on study-level covariates and multivariate
  meta-analysis of correlated effects; shrinkage of effects across conditions or
  studies, as mash does [@urbut2019]; the r-values of a primary study and its follow-up
  [@heller2014]; metap's methods of Edgington, Lancaster, the inverse t, the mean z and
  vote counting; weights for the studies of SumRank, and its permutations of designs
  with random effects; the proportion of studies that SumRank sums, chosen by
  cross-validation of held-out prediction as the paper chose it (a loop over the
  proportions does it today); cell types aligned up the Cell Ontology.
- **Questions not answered yet.**
  - Grouping patients by their profiles, such as by a factorization of samples across
    cell types, and predicting a continuous outcome from them, such as the severity of
    a disease, where held-out prediction tells cases from controls.
  - Which cell-cell interactions change between conditions, by ligand-receptor scoring
    and tests of differential interactions. @dimitrov2022 compared 16 resources and 7
    methods for cell-cell communication, and found that the choice of both strongly
    changes the predicted interactions; @dimitrov2024 introduced LIANA+, a framework
    for cell-cell communication in single-cell and spatial data, from one or several
    conditions.
  - How cells respond to drugs and doses: dose-response fits, benchmark doses and drug
    synergy. @srivatsan2020 introduced sci-Plex and profiled about 650,000 single cells
    across about 5,000 samples of three cancer cell lines exposed to 188 compounds.
  - Effects that depend on genotype are left to tools that map expression quantitative
    trait loci. @yazar2022 profiled 1,267,758 blood cells of 982 donors and mapped
    26,597 independent cis-expression quantitative trait loci across 14 cell types.

The chapters begin with [Pseudobulk profiles and cubes](cubes.md).
