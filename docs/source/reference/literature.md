---
description: A short reading list on sample-level statistics for single-cell and bulk data, grouped by the questions studies ask, each paper with a line on what it shows.
---

# Literature

A reading list grouped by the questions that studies of many samples ask
([Overview](../index.md)). Each line says what the paper shows, in its own terms, and
numbers are those the papers report. The methods behind CytePool's functions are cited
on the pages of [How the statistics work](../statistics/choosing_a_method.md); every
work is in the [Bibliography](bibliography.md).

## Cells or samples

- @squair2021 compared differential expression methods on 18 datasets with matched bulk
  RNA-seq. Methods that ignore the variation between biological replicates were biased
  and prone to false discoveries, finding hundreds of differentially expressed genes
  where there were no biological differences.
- @zimmerman2021 showed that cells of one individual are not independent, and that
  ignoring it inflates the type 1 error. They proposed mixed models with a random effect
  for the individual, and found pseudobulk methods conservative in comparison.
- @murphy2022 reran that benchmark with a measure that balances false positives and
  false negatives, and found that pseudobulk methods performed best.
- @junttila2022 compared 18 methods for differential states in multi-subject data:
  pseudobulk methods performed generally best, and methods that treat cells as
  independent gave many false positives.
- @crowell2020 surveyed cell-level mixed models and pseudobulk methods for differential
  state analysis across samples and conditions, with a simulation of multi-sample
  single-cell data (muscat).
- @lun2017 showed that ignoring plate effects loses control of the type I error, and
  that summing the counts of the cells of each plate restores it, without losing power
  in their simulations.
- @hafner2025 found that methods built for single cells offered no advantage over
  pseudobulk DESeq2 on individual datasets in nested designs, and suggested dream as a
  compromise at atlas scale.
- @nguyen2023 benchmarked 46 workflows for data from several batches: batch-corrected
  data rarely improved the analysis of sparse data, while modelling the batch as a
  covariate helped where batch effects were large.
- @heumos2023 summarized independent benchmarks into best-practice workflows for
  single-cell analysis.

## Which genes change, and do they replicate

- @kang2018 pooled cells of 8 lupus patients, split into interferon-beta stimulated and
  untreated cultures, and assigned cells to donors by their genotypes; the documentation
  uses these data ([Datasets](datasets.md)).
- @perez2022 profiled more than 1.2 million blood cells of 162 lupus cases and 99
  controls, and found interferon-stimulated genes raised in monocytes and fewer naive
  CD4 T cells.
- @murphy2023 re-analysed the first single-nucleus study of Alzheimer's disease: testing
  donors rather than cells gave 26 differentially expressed genes at a false discovery
  rate of 0.05, against 14,274, 549 times fewer.
- @nakatsuka2025 found that over 85% of the Alzheimer's disease genes called in one
  dataset were not reproduced in any of 16 others, and that a meta-analysis of ranks
  across datasets found genes that predicted disease better.
- @garg2021 re-analysed COVID-19 single-cell studies in a standard way and confirmed 8
  of 20 published results across datasets.

## Which cell types expand or contract

- @phipson2022 introduced propeller, which tests transformed cell-type proportions with
  linear models and empirical Bayes moderation across replicates.
- @buttner2021 introduced scCODA, a Bayesian model of compositional single-cell data.
- @mangiola2023 introduced sccomp, which models composition and its variability
  together, and found that the variability of proportions depends on their mean in every
  technology examined.
- @hoffman2026b introduced crumblr, precision-weighted mixed models of cell-type
  composition that test several levels of the cell lineage at once.
- @dann2022 introduced Milo, which tests differential abundance in neighbourhoods of a
  k-nearest neighbour graph instead of clusters.
- @yi2024 benchmarked six differential abundance methods for condition-specific cell
  states, and gave recommendations by dataset.
- @stephenson2021 profiled over 780,000 blood cells of 130 COVID-19 patients and related
  the expansion of cell populations to disease severity.

## Which pathways and regulators respond

- @wu2012 showed that gene set tests that permute genes are very sensitive to the
  correlation between genes, and introduced camera, which allows for it.
- @wu2010 introduced ROAST, which rotates the residuals of a linear model instead of
  permuting samples, so that it applies to almost any design.
- @subramanian2005 introduced gene set enrichment analysis.
- @holland2020 found that tools for pathway and transcription-factor activity built for
  bulk data, with curated gene sets, work on single-cell data, and that their results
  depend more on the gene sets than on the statistic.
- @badiaimompel2022 collected activity methods in one framework, decoupleR, and found
  simple linear models among the best at predicting perturbed regulators.

## What drives variation between donors

- @hoffman2016 introduced variancePartition, which uses linear mixed models to attribute
  the variation of each gene to disease, sex, tissue, genetics or technical factors.
- @hoffman2021 introduced dream, mixed models for repeated measures designs, and showed
  that other methods can give reproducible false positives driven by the genetic
  regulation of expression.
- @hoffman2026a introduced dreamlet, which runs precision-weighted mixed models on the
  pseudobulk profiles of every cell type, and applied it to 1.4 million nuclei from 150
  Alzheimer's disease cases and 149 controls.
- @yazar2022 profiled 1,267,758 blood cells of 982 donors and mapped 26,597 independent
  cis-expression quantitative trait loci across 14 cell types.

## Which interactions change

- @dimitrov2022 compared 16 resources and 7 methods for cell-cell communication, and
  found that the choice of both strongly changes the predicted interactions.
- @dimitrov2024 introduced LIANA+, a framework for cell-cell communication in
  single-cell and spatial data, from one or several conditions.

## How cells respond to drugs

- @srivatsan2020 introduced sci-Plex and profiled about 650,000 single cells across
  about 5,000 samples of three cancer cell lines exposed to 188 compounds.

## Combining studies

- @czi2025 described CZ CELLxGENE Discover, a curated corpus of over 93 million unique
  cells with standardized metadata.
- @sikkema2023 integrated 49 datasets into the Human Lung Cell Atlas, over 2.4 million
  cells from 486 individuals.
- @urbut2019 introduced mash, which estimates many effects across many conditions while
  allowing for correlations between them.
- @viechtbauer2010 described metafor, the R package for meta-analysis that CytePool's
  meta-analysis follows.
